@echo off
REM Cleanup script for math-prof-ai before committing

echo Cleaning up repository files...

REM Remove Python cache files
echo Removing __pycache__ directories...
rmdir /s /q __pycache__ 2>nul
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
del /s /q *.pyc 2>nul

REM Remove temporary files
echo Removing temporary files...
del /s /q *.tmp 2>nul
del /s /q *.bak 2>nul
del /s /q *.swp 2>nul

REM Remove logs (except the main log file if needed)
echo Cleaning logs directory...
mkdir logs 2>nul
del /q logs\config.log 2>nul

REM Remove test files that are no longer needed
echo Removing unnecessary test files...
del /q test_*.py 2>nul
del /q debug_*.py 2>nul
del /q *_fix.py 2>nul
del /q fix_*.py 2>nul
del /q *benchmark*.py 2>nul
del /q *_benchmark*.json 2>nul
del /q *_benchmark*.html 2>nul

REM Keep essential test files
echo. > .keep_test_files
git checkout -- test_connections.py
git checkout -- test_api_connections.py

REM Remove older versions of the main agent
echo Removing older agent versions...
del /q math_agent.py 2>nul
del /q math_agent_enhanced.py 2>nul
del /q math_agent.py.* 2>nul
del /q math_agent_simplified.py 2>nul

REM Remove JSON test results
echo Removing test result JSON files...
del /q *_test_results.json 2>nul
del /q *_results.json 2>nul

REM Remove test markdown files
echo Removing benchmark analysis files...
del /q benchmark_*.md 2>nul
del /q *benchmark_analysis.md 2>nul

echo Cleanup complete!
echo The following core files are retained:
echo - math_agent_langgraph.py (main application)
echo - load_config.py (configuration loader)
echo - config.toml (configuration file)
echo - requirements.txt (dependencies)
echo - setup_env.sh and .bat (environment setup)
echo - README.md (documentation)
echo - .env (environment variables - ensure this is in .gitignore)

echo Remember to review changes before committing! 