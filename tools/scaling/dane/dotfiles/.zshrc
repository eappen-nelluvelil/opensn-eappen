[[ -o interactive ]] || return

export SCRATCH=${SCRATCH:-/usr/workspace/nelluvelil1}
typeset -g OPENSN_PLATFORM=

_opensn_detect_platform()
{
  local host_name=${$(hostname -s):l}
  case $host_name in
    dane*) OPENSN_PLATFORM=dane ;;
    tuolumne*) OPENSN_PLATFORM=tuolumne ;;
    *) OPENSN_PLATFORM= ;;
  esac
}

_opensn_load_modules()
{
  module purge
  case $OPENSN_PLATFORM in
    dane)
      module load \
        python/3.13.2 \
        git/2.46.2 \
        cmake/3.30.5 \
        clang/19.1.3-magic \
        openmpi/4.1.2
      export CC=clang CXX=clang++ OMPI_CC=clang OMPI_CXX=clang++
      unset MPICH_GPU_SUPPORT_ENABLED MPICH_SMP_SINGLE_COPY_MODE
      ;;
    tuolumne)
      module load \
        cmake/3.29.2 \
        rocm/7.2.1 \
        rocmcc/7.2.1-magic \
        cray-mpich/9.1.0
      export CC=amdclang CXX=amdclang++
      export MPICH_GPU_SUPPORT_ENABLED=1
      export MPICH_SMP_SINGLE_COPY_MODE=XPMEM
      unset OMPI_CC OMPI_CXX
      ;;
    *)
      return
      ;;
  esac
}

_opensn_environment_file()
{
  case $OPENSN_PLATFORM in
    dane)
      print -r -- ${OPENSN_DANE_ENV:-$SCRATCH/opensn-dane/cbc-cycles-update/env.zsh}
      ;;
    tuolumne)
      print -r -- ${OPENSN_TUO_ENV:-$SCRATCH/opensn-tuo/current/env.zsh}
      ;;
  esac
}

opensn_env_reload()
{
  _opensn_detect_platform
  local env_file=$(_opensn_environment_file)
  if [[ -n $env_file && -r $env_file ]]; then
    source "$env_file"
  elif [[ -n $OPENSN_PLATFORM ]]; then
    _opensn_load_modules
  fi
  rehash
}

opensn_env_reload

HISTFILE=$HOME/.zsh_history
HISTSIZE=200000
SAVEHIST=200000
setopt APPEND_HISTORY
setopt AUTO_CD
setopt EXTENDED_HISTORY
setopt HIST_EXPIRE_DUPS_FIRST
setopt HIST_IGNORE_ALL_DUPS
setopt HIST_IGNORE_SPACE
setopt INTERACTIVE_COMMENTS
setopt SHARE_HISTORY

alias cp='cp -i'
alias mv='mv -i'
alias rm='rm -i'
alias ls='ls --color=auto'
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias grep='grep --color=auto'
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'
alias reload='exec zsh -l'
alias scratch='cd "$SCRATCH"'
alias tuo='cd "$SCRATCH/opensn-gpu"'
alias dane='cd "$SCRATCH/opensn-dane"'

export CDPATH=.:$SCRATCH

mkcd()
{
  [[ $# -eq 1 ]] || { print -u2 'usage: mkcd DIRECTORY'; return 2; }
  mkdir -p -- "$1" && cd -- "$1"
}

autoload -Uz compinit
compinit -d "$HOME/.zcompdump"

autoload -Uz colors
colors
setopt PROMPT_SUBST
PROMPT='%F{8}[%*]%f %F{2}%n@%m%f:%F{4}%~%f %F{3}$(git branch --show-current 2>/dev/null)%f
%# '
