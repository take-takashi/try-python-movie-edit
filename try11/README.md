# try11

Nuiktaでパッケージ化しようとしたけど、エラーが出ている。

## setup command

```bash
% poetry -V
Poetry (version 2.0.1)
% poetry init

This command will guide you through creating your pyproject.toml config.

Package name [try11]:  
Version [0.1.0]:  
Description []:  
Author [take-takashi <take.t.public@gmail.com>, n to skip]:  
License []:  MIT
Compatible Python versions [>=3.13]:  >=3.13,<3.14

Would you like to define your main dependencies interactively? (yes/no) [yes] no
Would you like to define your development dependencies interactively? (yes/no) [yes] no
Generated file

[project]
name = "try11"
version = "0.1.0"
description = ""
authors = [
    {name = "take-takashi",email = "take.t.public@gmail.com"}
]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.13,<3.14"
dependencies = [
]


[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"


Do you confirm generation? (yes/no) [yes] 

% pyenv local 3.13.1
% poetry env use $(pyenv which python)
Creating virtualenv try11 in /Users/takashi/github/try-python-movie-edit/try11/.venv
Using virtualenv: /Users/takashi/github/try-python-movie-edit/try11/.venv
poetry env info | grep -A 5 "Virtualenv" | grep "Python:" | awk '{print $2}'
3.13.1
% poetry shell
Spawning shell within /Users/takashi/github/try-python-movie-edit/try11/.venv
takashi@Mac try11 % emulate bash -c '. /Users/takashi/github/try-python-movie-edit/try11/.venv/bin/activate'
% deactivate
```

## poetry add

```bash
% poetry add ultralytics opencv-python torch torchvision ffmpeg-python "numpy<=2.1.1"
% poetry add -D pyinstaller taskipy nuitka
```

## pyproject.toml setting

```bash
% echo -n '
[tool.taskipy.tasks]
build = "pyinstaller --onefile --clean --add-data 'assets:assets' main.py"
' >> pyproject.toml
```
