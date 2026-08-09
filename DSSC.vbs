Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\userapp\Projects\DeepSeaSedimentClassifier"
WshShell.Run """C:\Python314\pythonw.exe"" main.py", 0, False
