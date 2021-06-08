WICHTIG:
Die IRIS Daten sowie der Kunde unterliegen einer starken Geheimhaltungsvereinbarung. Verwendet die Daten nur für interne Zwecke solltet ihr Präsentationsmaterial benötigen oder ähnliches sprecht das bitte vorher mit Stephanie Wegner ab.


Als erstes wird Python als Programmiersprache benötigt:

1. Download Python via https://www.python.org/downloads/ (Version 3.9.5)
2. Installieren von Python mittels der heruntergeladenen Datei
   Eventuell auswählen der Option: "Add Python to PATH" (oder ähnlich)
3. Python verwendet als Package-Manager pip. Dieser wird automatisch mit installiert. Damit alle die gleichen Packages und vor allem die gleichen Versionen davon verwenden, legt man am besten eine Umgebung an, in der dann gearbeitet wird:

1. Installieren des Packages virtualenv über das Terminal mit: python3 -m pip install virtualenv
   mehr Infos unter https://virtualenv.pypa.io/en/stable/
2. Installieren eines Wrappers zur einfacheren Handhabe über das Terminal mit: python3 -m pip install virtualenvwrapper
   mehr Infos unter https://virtualenvwrapper.readthedocs.io/en/latest/install.html
3. Um diesen nutzen zu können muss noch folgendes eingegeben werden:
   export VIRTUALENVWRAPPER_PYTHON=/usr/local/bin/python3
4. Anschließend muss folgendes ausgeführt werden:
   source /Library/Frameworks/Python.framework/Versions/3.9/bin/virtualenvwrapper.sh (Der Dateipfad kann ggf. variieren und ist über which virtualenvwrapper.sh einsehbar )
5. Nun kann eine Umgebung erstellt werden: mkvirtualenv `<name>` (optional noch hinzufügen -a `<Pfad>`)
   Um nun in der Umgebung arbeiten zu können, wird folgender Befehl verwendet: workon `<name>
   Um eine Umgebung wieder zu verlassen, wird der Befehl deactivate verwendet.
6. Eine Liste der erstellten Umgebungen kann mit lsvirtualenv angezeigt werden.
7. Um eine Umgebung wieder zu löschen, gibt es folgenden Befehl: rmvirtualenv `<name>`


Da nun die Umgebung erzeugt wurde, fehlen jetzt nur noch die benötigten Packages, um damit arbeiten zu können. Zu jedem Projekt sollte eine Requirement-Datei existieren, in der alle Packages mit den richtigen Versionen gespeichert sind. Um diese zu installieren, muss in den jeweiligen Ordner navigiert werden und dort folgender Befehl ausgeführt werden (während die Umgebung aktiv ist): pip install -r requirements.txt



Projekteinbindung (Intellij):

Repository klonen
Projekt mit Intellij öffnen
An diesem Punkt sollte Intellij vorschlagen, dass das Python Plugin installiert werden soll

Anschließend muss Intellij neugestartet werden
Nach dem Neustart muss die SDK noch gesetzt werden, dies schlägt Intellij ebenfalls automatisch vor, falls eine .py-Datei geöffnet ist

---

Folgendes Setup ist zu empfehlen, wenn Python intensiv genutz wird und verschiedene Python Versionen zum Einsatz kommen könnten.
Download verschiedener Python Versionen und deren Verwaltung kann nervig werden. Dazu bietet sich `pyenv` an (Python package manager).
1. Installieren pyenv https://github.com/pyenv/pyenv
   1.1 z.B `brew install pyenv`
   Um am Ende nicht in einer Dependency-Hölle zu landen bietet sich außerdem `pipenv` an (siehe https://realpython.com/pipenv-guide/).
   Unter MAC BIG SUR 11.3.1: Folgendes zur .zshrc hinzufügen:
   ```zsh
       export PYENV_ROOT="$HOME/.pyenv"
       export PATH="$PYENV_ROOT/bin:$PATH"
       export PATH="$PYENV_ROOT/shims:$PATH"
       if command -v pyenv 1>/dev/null 2>&1; then
       eval "$(pyenv init -)"
       fi
   ```
2. Nach Installation der gewünschten python-version über pyenv, einfach `pip install pipenv`
3. Dann im Projektordner `pipenv install` (Installiert die Dependencies aus dem Pipfile in eine eigene virtualenv.)

Hinweis: Falls im Moment Tensorflow verwendet werden soll, muss python 3.8 verwendet werden. (19.05.21)
Initales Pipfile mit typischen Machine-learning bibliotheken ist beigefügt.


MacOs mit alten Term:
```
.zshrc:
export SYSTEM_VERSION_COMPAT=1
```
