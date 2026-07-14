{ pkgs, ... }:
{
  languages.rust.enable = true;
  languages.rust.channel = "stable";

  packages = with pkgs; [
    rust-analyzer
  ];

  enterShell = ''
    echo "llm-inference: Rust $(rustc --version)"
  '';
}
