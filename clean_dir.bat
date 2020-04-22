@REM @Author: lpereira
@REM @Date:   2019-09-09 16:02:34
@REM @Last Modified by:   lpereira
@REM Modified time: 2020-04-21 22:08:15

@REM see: https://stackoverflow.com/a/53341473

ERASE /S *.rpy *.rpy.* *.pyc *.abq *.com *.dat *.mdl *.msg *.pac *.res *.sel *.stt *.rec *.dmp *.log
FOR /d /r . %%d IN ("__pycache__") DO @IF EXIST "%%d" rmdir /s /q "%%d"
