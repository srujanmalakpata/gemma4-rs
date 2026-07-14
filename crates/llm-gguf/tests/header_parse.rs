use llm_gguf::reader::GgufFile;
use std::io::Cursor;

#[test]
fn parses_magic_version_and_tensor_count_from_minimal_file() {
    let bytes = llm_gguf::testutil::write_minimal_q8_tensor("demo.weight", &[1; 32]);
    let file = GgufFile::open_read(Cursor::new(bytes)).unwrap();

    assert_eq!(file.tensor_names(), ["demo.weight"]);
    assert_eq!(file.tensor("demo.weight").unwrap().numel(), 32);
}
