import subprocess

def get_package_version(package_name):
    try:
        result = subprocess.run(['pip', 'show', package_name], capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        version_line = next(line for line in lines if line.startswith('Version:'))
        version = version_line.split(':', 1)[1].strip()
        return f"{package_name}: {version}"
    except subprocess.CalledProcessError as e:
        return f"Error getting info for package {package_name}: {e}"
    except StopIteration:
        return f"Version info not found for package {package_name}"

packages = [
    'streamlit',
    'tensorflow',
    'numpy',
    'pandas',
    'matplotlib',
    'streamlit-option-menu',
    'keras-tuner',
    'scikit-learn',
    'seaborn'
]

with open('package_info.txt', 'w') as f:
    for package in packages:
        info = get_package_version(package)
        f.write(info + '\n')

print("Package information has been written to package_info.txt")
