//! Worker threads for row-parallel mat-vecs, started once and reused.
//!
//! Starting threads for every mat-vec (as `std::thread::scope` does) cost more
//! than the work itself on most of Gemma's matrices: a profile of E2B decoding
//! spent a large share of its samples creating and tearing down threads.
//!
//! Each mat-vec becomes one task, sent to every worker. The caller and the
//! workers then claim chunks of rows through an atomic counter until none are
//! left, so faster cores take more chunks and nobody queues behind a lock.
//! Tasks own their inputs through `Arc`, so the pool needs no `unsafe`.
//!
//! While a model is decoding, a new mat-vec arrives every few tens of
//! microseconds, so workers and the caller spin for a short while before
//! sleeping: putting a thread to sleep and waking it again costs more.

use std::{
    ops::Range,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, TryRecvError},
        Arc, Condvar, Mutex, OnceLock,
    },
    time::{Duration, Instant},
};

/// How long an idle worker, or a caller waiting for the last chunks,
/// spins before sleeping.
const SPIN_FOR: Duration = Duration::from_micros(200);

struct Pool {
    workers: Vec<mpsc::Sender<Arc<Task>>>,
}

/// Rows are split into this many chunks per thread, so faster cores take
/// more chunks instead of waiting for slower ones.
const CHUNKS_PER_THREAD: usize = 4;

struct Task {
    chunks: usize,
    next: AtomicUsize,
    remaining: AtomicUsize,
    run_chunk: Box<dyn Fn(usize) + Send + Sync>,
    finished: Mutex<bool>,
    finished_signal: Condvar,
}

impl Task {
    /// Claims and runs chunks until none are left.
    fn work(&self) {
        loop {
            let chunk = self.next.fetch_add(1, Ordering::Relaxed);
            if chunk >= self.chunks {
                return;
            }
            // A panicking chunk still counts as finished, so the caller never
            // waits forever; its empty result slot is reported instead.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.run_chunk)(chunk)));
            if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                *self.finished.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
                self.finished_signal.notify_all();
            }
        }
    }

    fn wait(&self) {
        let deadline = Instant::now() + SPIN_FOR;
        while Instant::now() < deadline {
            if self.remaining.load(Ordering::Acquire) == 0 {
                return;
            }
            std::hint::spin_loop();
        }
        let mut finished = self.finished.lock().unwrap_or_else(|poison| poison.into_inner());
        while !*finished {
            finished = self.finished_signal.wait(finished).unwrap_or_else(|poison| poison.into_inner());
        }
    }
}

/// Number of threads a mat-vec uses, the caller included: `LLM_THREADS` if
/// set, otherwise every core the OS reports.
pub fn threads() -> usize {
    pool().workers.len() + 1
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::env::var("LLM_THREADS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&count: &usize| count > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
        let workers = (1..threads)
            .map(|index| {
                let (sender, tasks) = mpsc::channel::<Arc<Task>>();
                std::thread::Builder::new()
                    .name(format!("matvec-{index}"))
                    .spawn(move || {
                        while let Some(task) = next_task(&tasks) {
                            task.work();
                        }
                    })
                    .expect("start a mat-vec worker thread");
                sender
            })
            .collect();
        Pool { workers }
    })
}

/// Waits for the next task, spinning for [`SPIN_FOR`] before sleeping.
/// Returns `None` once the pool is gone.
fn next_task(tasks: &mpsc::Receiver<Arc<Task>>) -> Option<Arc<Task>> {
    let deadline = Instant::now() + SPIN_FOR;
    loop {
        match tasks.try_recv() {
            Ok(task) => return Some(task),
            Err(TryRecvError::Disconnected) => return None,
            Err(TryRecvError::Empty) if Instant::now() < deadline => std::hint::spin_loop(),
            Err(TryRecvError::Empty) => return tasks.recv().ok(),
        }
    }
}

/// Evaluates `value(&data, i)` for `i` in `0..count`. Runs on the calling
/// thread when the total work (`count * work_per_item`) is below
/// `parallel_min_work`, otherwise in chunks on the calling thread and the
/// workers.
pub fn map_indices<D: Send + Sync + 'static>(
    data: Arc<D>,
    count: usize,
    work_per_item: usize,
    parallel_min_work: usize,
    value: fn(&D, usize) -> f32,
) -> Vec<f32> {
    run_chunks(count, work_per_item, parallel_min_work, move |range| {
        range.map(|index| value(&data, index)).collect()
    })
}

/// Like [`map_indices`] for items that produce several values each (for
/// example one attention head): concatenates `value(&data, i)` for `i` in
/// `0..count`.
pub fn map_items<D: Send + Sync + 'static>(
    data: Arc<D>,
    count: usize,
    work_per_item: usize,
    parallel_min_work: usize,
    value: fn(&D, usize) -> Vec<f32>,
) -> Vec<f32> {
    run_chunks(count, work_per_item, parallel_min_work, move |range| {
        range.flat_map(|index| value(&data, index)).collect()
    })
}

/// Splits `0..count` into chunks and concatenates `run(chunk)` in order,
/// spreading chunks over the calling thread and the workers when the work
/// is large enough.
fn run_chunks(
    count: usize,
    work_per_item: usize,
    parallel_min_work: usize,
    run: impl Fn(Range<usize>) -> Vec<f32> + Send + Sync + 'static,
) -> Vec<f32> {
    let pool = pool();
    if pool.workers.is_empty() || count.saturating_mul(work_per_item) < parallel_min_work {
        return run(0..count);
    }
    let chunk_len = count.div_ceil(threads() * CHUNKS_PER_THREAD).max(1);
    let chunks = count.div_ceil(chunk_len);
    let slots: Arc<Vec<OnceLock<Vec<f32>>>> = Arc::new((0..chunks).map(|_| OnceLock::new()).collect());
    let task = Arc::new(Task {
        chunks,
        next: AtomicUsize::new(0),
        remaining: AtomicUsize::new(chunks),
        run_chunk: {
            let slots = Arc::clone(&slots);
            Box::new(move |chunk| {
                let start = chunk * chunk_len;
                // Each chunk is claimed exactly once, so its slot is empty.
                let _ = slots[chunk].set(run(start..(start + chunk_len).min(count)));
            })
        },
        finished: Mutex::new(false),
        finished_signal: Condvar::new(),
    });
    for worker in &pool.workers {
        worker.send(Arc::clone(&task)).expect("mat-vec worker threads are running");
    }
    task.work();
    task.wait();
    slots
        .iter()
        .flat_map(|slot| slot.get().expect("every chunk ran").iter().copied())
        .collect()
}
