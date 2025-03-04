# try12

`ffmpeg-python`が実行するPCにffmpegをインストールしていないと使えない模様なので
音声の合成をOpenCVで実施したい。

## setup command

```bash
% poetry -V
Poetry (version 2.0.1)
% poetry init
takashi@Mac try12 % pyenv local 3.13.1
takashi@Mac try12 % poetry env use $(pyenv which python)
Creating virtualenv try12 in /Users/takashi/github/try-python-movie-edit/try12/.venv
Using virtualenv: /Users/takashi/github/try-python-movie-edit/try12/.venv
takashi@Mac try12 % poetry env info | grep -A 5 "Virtualenv" | grep "Python:" | awk '{print $2}'
3.13.1
takashi@Mac try12 % poetry shell
Spawning shell within /Users/takashi/github/try-python-movie-edit/try12/.venv
takashi@Mac try12 % emulate bash -c '. /Users/takashi/github/try-python-movie-edit/try12/.venv/bin/activate'
(try12-py3.13) takashi@Mac try12 % deactivate
```
