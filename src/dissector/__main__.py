"""
Entry point for running dissector as a module: python -m dissector
"""
import os
import platform
import multiprocessing
import uvicorn


def main():
    print("Starting Dissector service...")
    print("Models will be loaded on startup...")
    
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('DEBUG', 'true').lower() == 'true'
    workers = int(os.getenv('WORKERS', 0))
    
    if debug:
        uvicorn.run("dissector.app:app", host='0.0.0.0', port=port, log_level='info', reload=True)
    else:
        if platform.system() == "Darwin":
            if workers > 1:
                print(f"Warning: macOS detected. Forcing workers=1 (was {workers}) to avoid MLX concurrency issues.")
            workers = 1
        elif workers <= 0:
            cpu_count = multiprocessing.cpu_count()
            workers = min(cpu_count, 4)
        
        uvicorn.run(
            "dissector.app:app",
            host='0.0.0.0',
            port=port,
            log_level='info',
            workers=workers,
            reload=False
        )


if __name__ == '__main__':
    main()

