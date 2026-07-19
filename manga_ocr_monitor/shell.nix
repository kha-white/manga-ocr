{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312;

  manga-ocr = python.pkgs.buildPythonPackage rec {
    pname = "manga-ocr";
    version = "0.1.11";

    src = python.pkgs.fetchPypi {
      inherit pname version;
      sha256 = "sha256-Ic6AaSaEY1NaPHgF9xawj1gmrwWqf1NhmTs8NYKZsOs=";
    };

    postPatch = ''
      sed -i \
        's/self(example_path)/if example_path.exists(): self(example_path)/' \
        manga_ocr/ocr.py
    '';

    propagatedBuildInputs = with python.pkgs; [
      pillow
      numpy
      torch
      torchvision
      transformers
      sentencepiece
      jaconv
      loguru
      fugashi
      mecab-python3
      unidic-lite
    ];

    doCheck = false;
  };

in
pkgs.mkShell {
  packages = [
    (python.withPackages (ps: with ps; [
      pillow
      numpy
      torch
      torchvision
      transformers
      sentencepiece
      manga-ocr
    ]))

    # Wayland
    pkgs.wl-clipboard

    # X11
    pkgs.xclip

    # Graphcs libs
    pkgs.libGL
    pkgs.mesa
    pkgs.libjpeg
    pkgs.libpng
    pkgs.freetype
    pkgs.zlib
    pkgs.libffi
    pkgs.pkg-config
  ];

  shellHook = ''
    export HF_HOME=$PWD/.hf-cache
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
    echo "MangaOCR ready (Wayland + X11)"
  '';
}

