use llm_gguf::reader::GgufFile;
use std::io::Cursor;

#[test]
fn parses_metadata_arrays_and_reads_q8_tensor_bytes() {
    let values = [1; 32];
    let bytes = write_gguf_with_metadata(&values);
    let mut file = GgufFile::open_read(Cursor::new(bytes)).unwrap();

    assert_eq!(file.tensor_names(), ["demo.weight"]);
    assert_eq!(file.tensor("demo.weight").unwrap().numel(), 32);
    assert_eq!(
        file.metadata_strings("tokenizer.ggml.tokens").unwrap(),
        ["<bos>", "hello"]
    );
    assert_eq!(
        file.metadata_f32s("tokenizer.ggml.scores").unwrap(),
        [0.0, -1.5]
    );
    assert_eq!(
        file.metadata_i32s("tokenizer.ggml.token_type").unwrap(),
        [1, 2]
    );

    let tensor = file.read_tensor_bytes("demo.weight").unwrap();
    assert_eq!(tensor.len(), 34);
    assert_eq!(&tensor[2..], values.map(|value| value as u8));
}

#[test]
fn unsupported_tensor_type_names_the_tensor() {
    let mut bytes = llm_gguf::testutil::write_minimal_q8_tensor("bad.weight", &[1; 32]);
    let tensor_type_offset = 4 + 4 + 8 + 8 + 8 + "bad.weight".len() + 4 + 8;
    bytes[tensor_type_offset..tensor_type_offset + 4].copy_from_slice(&999_u32.to_le_bytes());

    let error = match GgufFile::open_read(Cursor::new(bytes)) {
        Ok(_) => panic!("unsupported tensor type should fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("bad.weight"));
}

fn write_gguf_with_metadata(values: &[i8; 32]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    write_u32(&mut bytes, 3);
    write_u64(&mut bytes, 1);
    write_u64(&mut bytes, 3);

    write_array_header(&mut bytes, "tokenizer.ggml.tokens", 8, 2);
    write_string(&mut bytes, "<bos>");
    write_string(&mut bytes, "hello");

    write_array_header(&mut bytes, "tokenizer.ggml.scores", 6, 2);
    bytes.extend_from_slice(&0.0_f32.to_le_bytes());
    bytes.extend_from_slice(&(-1.5_f32).to_le_bytes());

    write_array_header(&mut bytes, "tokenizer.ggml.token_type", 5, 2);
    bytes.extend_from_slice(&1_i32.to_le_bytes());
    bytes.extend_from_slice(&2_i32.to_le_bytes());

    write_string(&mut bytes, "demo.weight");
    write_u32(&mut bytes, 1);
    write_u64(&mut bytes, 32);
    write_u32(&mut bytes, 8);
    write_u64(&mut bytes, 0);

    bytes.resize(bytes.len().next_multiple_of(32), 0);
    bytes.extend_from_slice(&0x3c00_u16.to_le_bytes());
    bytes.extend(values.iter().map(|&value| value as u8));
    bytes
}

fn write_array_header(bytes: &mut Vec<u8>, key: &str, element_type: u32, count: u64) {
    write_string(bytes, key);
    write_u32(bytes, 9);
    write_u32(bytes, element_type);
    write_u64(bytes, count);
}

fn write_string(bytes: &mut Vec<u8>, value: &str) {
    write_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
