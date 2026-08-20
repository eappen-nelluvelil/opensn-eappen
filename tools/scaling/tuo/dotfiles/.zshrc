[[ -o interactive ]] || return

_opensn_initialize_modules()
{
  (( $+functions[module] || $+commands[module] )) && return
  local candidate
  for candidate in \
    ${MODULESHOME:-}/init/zsh \
    /usr/share/lmod/lmod/init/zsh \
    /etc/profile.d/z00_lmod.sh
  do
    if [[ -n $candidate && -r $candidate ]]; then
      source "$candidate"
      return
    fi
  done
}

opensn_modules()
{
  _opensn_initialize_modules
  module purge
  case ${$(hostname -s):l} in
    dane*)
      module load python/3.13.2 git/2.46.2 cmake/3.30.5 clang/19.1.3-magic openmpi/4.1.2
      export CC=clang CXX=clang++ OMPI_CC=clang OMPI_CXX=clang++
      unset MPICH_GPU_SUPPORT_ENABLED MPICH_SMP_SINGLE_COPY_MODE
      ;;
    tuolumne*)
      module load python/3.13.2 cmake/3.29.2 rocm/7.2.1 rocmcc/7.2.1-magic cray-mpich/9.1.0
      export CC=amdclang CXX=amdclang++
      export MPICH_GPU_SUPPORT_ENABLED=1
      export MPICH_SMP_SINGLE_COPY_MODE=XPMEM
      unset OMPI_CC OMPI_CXX
      ;;
  esac
  rehash
}

opensn_use()
{
  [[ $# -eq 1 && -r $1 ]] || {
    print -u2 'usage: opensn_use /path/to/env.zsh'
    return 2
  }
  source "$1"
  rehash
}

opensn_modules

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
