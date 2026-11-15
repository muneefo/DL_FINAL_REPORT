# Recovery after a Windows restart

Training saves a resume checkpoint every 1,000 optimizer updates (approximately
15–17 minutes at the recently observed speed). It includes model weights, both
Adam optimizer states, scheduler state, completed update count, loss history,
and CPU/CUDA random states. A temporary file is written before atomically
replacing the checkpoint, so interruption during a save retains the prior file.

After restarting Windows, run from this project folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\Resume-Project.ps1
```

The launcher checks for an existing trainer/coordinator to avoid duplicates.
The queue skips completed jobs and resumes interrupted training using its saved
checkpoint. Unsaved updates since the last checkpoint are repeated. Do not pass
an old process ID after reboot. Nothing has been installed into Windows Startup;
this launcher must be run after login, or the assistant can run it on request.

Current cohort: `course_completion_20261109-133839`.
Latest training progress and checkpoint files are in its
`stage2_fraction_models/` directory. Leave that directory and the saved
configuration/source files intact.
