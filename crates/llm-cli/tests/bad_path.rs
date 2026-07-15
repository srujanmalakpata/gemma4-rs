use std::process::Command;

#[test]
fn missing_model_path_fails_cleanly() {
    let output = Command::new(env!("CARGO_BIN_EXE_llm-cli"))
        .args(["--model", "models/not-present.gguf", "--prompt", "hi"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("models/not-present.gguf"));
}
