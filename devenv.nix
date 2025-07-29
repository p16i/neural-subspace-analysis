{ pkgs, lib, config, ... }: {
  languages.python = {
    enable = true;
    version = "3.12";
    poetry.enable = true;
    poetry.activate.enable = true;
  };

  git-hooks.hooks.check-case-conflicts.enable = true;
  git-hooks.hooks.check-merge-conflicts.enable = true;
  git-hooks.hooks.check-toml.enable = true;
  git-hooks.hooks.check-yaml.enable = true;
  git-hooks.hooks.trim-trailing-whitespace.enable = true;

  git-hooks.hooks.ruff.enable = true;
  git-hooks.hooks.ruff-format.enable = true;
}
