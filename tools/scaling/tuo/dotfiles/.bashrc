# Bash fallback. Non-interactive commands must remain silent.

case $- in
  *i*) ;;
  *) return ;;
esac

if [[ -x /bin/zsh ]]; then
  exec /bin/zsh -l
fi
