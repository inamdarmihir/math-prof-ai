@echo off
echo Installing benchmark requirements...
pip install -r requirements_benchmark.txt

echo Running JEE Benchmark...
python jee_benchmark.py --max-problems 5

echo.
echo Benchmark completed! Opening report...
start jee_benchmark_results.html

echo.
echo To run the full benchmark, use: python jee_benchmark.py 