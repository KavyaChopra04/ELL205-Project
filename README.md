Clone the repo on your local machine


On Ubuntu, run:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python3 Stabilization.py <path_to_video>

```
(Create a virtual environment to install dependencies, and then install those and run the file)

On Windows, run:

First, open Powershell with admin permissions and run 
```Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force```
Then, open the cloned repository in VS Code and run:

```
python3 -m venv env
./env/Scripts/activate
pip install -r requirements.txt
python Stabilization.py <path_to_video>

```

For any doubt in running/setting up the project, please mail me at my IITD email address 


Acknowledgements:
https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization/tree/master
http://www.liushuaicheng.org/eccv2016/meshflow.pdf
