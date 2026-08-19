# Login-wide settings shared by Dane and Tuolumne. Keep this file silent.

umask 077
export EDITOR=vi
export VISUAL=$EDITOR
export SCRATCH=/usr/workspace/nelluvelil1

[[ -d $HOME/bin ]] && path=($HOME/bin $path)
[[ -d $HOME/.local/bin ]] && path=($HOME/.local/bin $path)
typeset -U path PATH
