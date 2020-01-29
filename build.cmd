@echo off
title qmake and nmake build prompt
set VCINSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC
set QTDIR=C:\Qt\Qt5.9.8\5.9.8\msvc2017_64
set PATH=%VCINSTALLDIR%\bin;%QTDIR%\bin;C:\Program Files\7-Zip;%PATH%
call "%VCINSTALLDIR%\vcvarsall.bat" amd64
qmake IPS.pro -spec win32-msvc "CONFIG+=debug" "CONFIG+=qml_debug"
nmake
cd debug
windeployqt IPS.exe
