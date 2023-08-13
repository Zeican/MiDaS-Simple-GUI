@Echo off
echo "Opening sub-directory 'dist'"
cd ./dist
echo "Installing Depth Estimation Wheel..."
python.exe -m pip install Depth_Estimation_Simple_GUI-1.0.0-cp310-cp310-win_amd64.whl
echo "Done!"
SET /p launch="Launch Depth Estimation GUI? (Y/N)"
IF /i "%launch%" == "y" GOTO yes
IF /i "%launch%" == "Y" GOTO yes
IF /i "%launch%" == "n" GOTO no
IF /i "%launch%" == "N" GOTO no


:yes
cls
echo "Launching..."
cd..
python depth_estimation.py
PAUSE


:no
cls
echo "Exiting..."
timeout /t 5 /nobreak
exit