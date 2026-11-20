{
  description = "Devenv.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };
  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        system = system;
        config.allowUnfree = true;
        # config.allowBroken = true;
        config.cudaSupport = true;
      };
      python = pkgs.python313.override {
        packageOverrides = new: old: {
          umap-learn = old.umap-learn.overridePythonAttrs (oldAttrs: {
            doCheck = false;
          });
          biopython = old.biopython.overridePythonAttrs (oldAttrs: {
            doCheck = false;
          });
          onnxscript = old.onnxscript.overridePythonAttrs (oldAttrs: {
            doCheck = false;
          });
        };
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell rec {
        buildInputs = with pkgs; [
          (python.withPackages (
            ps: with ps; [
              biopython
              datasets
              dash
              polars
              openai
              hydra-core
              faiss
              gradio
              tabulate
              ipdb
              joblib
              kaleido
              lightning
              matplotlib
              networkx
              nltk
              omegaconf
              onnxruntime
              onnxscript
              pandas
              pip
              pip-tools
              plotly
              pudb
              pytest
              sentence-transformers
              sentence-transformers.optional-dependencies.image
              seqeval
              torch-geometric
              transformers
              torchvision
              umap-learn
              wandb

              (buildPythonPackage rec {
                pname = "pytrec-eval";
                version = "0.5";
                format = "setuptools";
                __structuredAttrs = true;

                src = fetchFromGitHub {
                  owner = "cvangysel";
                  repo = "pytrec_eval";
                  tag = version;
                  hash = "sha256-t76D3C5QMJgQMhAg8TGxdtjwaLQhlB8SufAdM3pAZg4=";
                  fetchSubmodules = true;
                };

                build-system = [
                  setuptools
                ];

                dependencies = [
                  numpy
                  scipy
                ];
                NIX_CFLAGS_COMPILE = "-std=gnu17 -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=incompatible-pointer-types";
              })

            ]
          ))
          cudaPackages.cuda_nvrtc.lib
          cudaPackages.cuda_nvrtc
          cudaPackages.cudnn
          cudaPackages.libcublas
          cudaPackages.libcufft
          cudaPackages.libcurand
        ];

        shellHook = ''
          unset SOURCE_DATE_EPOCH
          export CUDA_PATH=${pkgs.cudatoolkit}
          export LD_LIBRARY_PATH=${
            pkgs.lib.makeLibraryPath (
              [
                "/run/opengl-driver" # Needed to find libGL.so
              ]
              ++ buildInputs
            )
          }:$LD_LIBRARY_PATH

          # Set LIBRARY_PATH to help the linker find the CUDA static libraries
          export LIBRARY_PATH=${
            pkgs.lib.makeLibraryPath [
              pkgs.cudatoolkit
            ]
          }:$LIBRARY_PATH
          export LD_PRELOAD=${pkgs.cudaPackages.cuda_nvrtc.lib}/lib/libnvrtc.so.12:$LD_PRELOAD
        '';
      };
    };
}
