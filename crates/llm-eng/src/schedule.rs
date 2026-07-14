pub enum Phase {
    Prefill,
    Decode { position: usize },
}

pub struct SingleSequence {
    pub max_context: usize,
    pub phase: Phase,
}
