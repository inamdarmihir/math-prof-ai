@echo off
:: Pre-commit hook to check for sensitive data before commits
:: Copy this file to .git\hooks\pre-commit.bat 

echo Running pre-commit checks...

:: Check for API keys in staged files
echo Checking for API keys in staged files...
git diff --cached --name-only > temp_staged_files.txt
findstr /m "sk-" temp_staged_files.txt > nul
if %ERRORLEVEL% EQU 0 (
  echo ERROR: Potential OpenAI API key found in staged files.
  echo Please remove API keys before committing.
  del temp_staged_files.txt
  exit /b 1
)

:: Check for .env files
echo Checking for .env files...
findstr /m "\.env$" temp_staged_files.txt > nul
if %ERRORLEVEL% EQU 0 (
  echo ERROR: .env file is staged for commit.
  echo Please remove .env files from staging before committing.
  del temp_staged_files.txt
  exit /b 1
)

:: Check for backup files
echo Checking for backup files...
findstr /m "\.(bak|backup|old|tmp|log)$" temp_staged_files.txt > nul
if %ERRORLEVEL% EQU 0 (
  echo WARNING: Backup or log files are staged for commit.
  set /p answer=Do you want to continue? (y/n)
  if /i not "%answer%"=="y" (
    del temp_staged_files.txt
    exit /b 1
  )
)

del temp_staged_files.txt
echo Pre-commit checks passed!
exit /b 0 