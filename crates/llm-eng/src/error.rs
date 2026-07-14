use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngError {
    #[error("context length {got} exceeds configured max {max}")]
    ContextTooLong { got: usize, max: usize },
    #[error("layer {0} has no cached K/V")]
    MissingLayer(usize),
    #[error("missing weight `{0}`")]
    MissingWeight(String),
    #[error("compute: {0}")]
    Compute(String),
    #[error("{0}")]
    Msg(String),
}
