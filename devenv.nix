{ pkgs, lib, config, ... }: {
  languages.python = {
    enable = true;
    version = "3.12";
    poetry.enable = true;
    poetry.activate.enable = true;
  };
}
