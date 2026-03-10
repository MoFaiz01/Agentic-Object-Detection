@echo off
cd /d g:\Agentic_Object_Detection_System
set PYTHONPATH=.
python -c "from deployment.app import app; app.run(host='127.0.0.1', port=5000, debug=False)"
