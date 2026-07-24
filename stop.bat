@echo off
echo Stopping AI News Monitor Services gracefully...
echo.

:: We use the /FI filter to match the exact window titles we set in run.bat
:: The /T switch kills any child processes, but we don't use /F (force) so they have a chance to shut down cleanly.
:: If you need a forced kill, you can add /F

echo Stopping Data Ingestion...
taskkill /FI "WINDOWTITLE eq Data Ingestion*" /T

echo Stopping Intelligence Pipeline...
taskkill /FI "WINDOWTITLE eq Intelligence Pipeline*" /T

echo Stopping API Backend...
taskkill /FI "WINDOWTITLE eq API Backend*" /T

echo Stopping Frontend Dashboard...
taskkill /FI "WINDOWTITLE eq Frontend Dashboard*" /T

echo Stopping Background Scheduler...
taskkill /FI "WINDOWTITLE eq Background Scheduler*" /T

echo Stopping Reprocessor...
taskkill /FI "WINDOWTITLE eq Reprocessor*" /T

echo.
echo All AI News Monitor services have been instructed to shut down.
echo If a window prompts you to terminate batch jobs, type 'Y'.
pause
