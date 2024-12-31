import pkg_resources

# List of dependencies you use
dependencies = [
    "dataclasses",
    "diffusers",
    "huggingface_hub",
    "lycoris",
    "torch",
]

# Get versions for each dependency
requirements = []
for package in dependencies:
    try:
        version = pkg_resources.get_distribution(package).version
        requirements.append(f"{package}=={version}")
    except pkg_resources.DistributionNotFound:
        print(f"Package {package} not found!")

# Write to requirements.txt
with open("requirements.txt", "w") as f:
    f.write("\n".join(requirements))

print("requirements.txt file created successfully!")
