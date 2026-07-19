{
  description = "Japanese OCR clipboard monitor";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
      in {
        packages.default = python.pkgs.buildPythonApplication {
          pname = "manag-ocr-monitor";
          version = "0.1.0";
          src = self;

          pyproject = true;
          build-system = with python.pkgs; [ setuptools ];

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
        };

        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/manga-ocr-monitor";
        };
      });
}
