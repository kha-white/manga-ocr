{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312;
in
pkgs.python312Packages.buildPythonApplication {
  pname = "manga-ocr-monitor";
  version = "0.1.0";

  src = ./.;

  pyproject = true;
  build-system = with pkgs.python312Packages; [ setuptools ];

  dontWrapPythonPrograms = false;
  dontUsePythonReexec = true;

  propagatedBuildInputs = with python.pkgs; [
    pillow
    numpy
    torch
    torchvision
    transformers
    sentencepiece
    jaconv
    fugashi
    mecab-python3
    unidic-lite
    loguru
    manga-ocr
  ] ++ [
    pkgs.wl-clipboard
    pkgs.xclip
  ];

  doCheck = false;
}
