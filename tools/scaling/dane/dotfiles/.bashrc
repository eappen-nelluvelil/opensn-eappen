# Bash fallback configuration. Non-interactive SSH commands must remain silent.

case $- in
  *i*) ;;
  *) return ;;
esac

PS1='[\t] \u@\h:\w\n\$ '
alias cp='cp -i'
alias mv='mv -i'
alias rm='rm -i'
