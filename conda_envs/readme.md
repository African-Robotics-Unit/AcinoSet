If you encounter PackageNotFound errors from the .yml files when creating the envs, you can omit the platform-specific build constraints from the env definition export

```sh
$ conda env export -n env_name -f env_name.yml --no-builds
```

If that doesn't work, run this script to remove the platform-specific build constraints:

```sh
$ sed 's/\(.*[[:alnum:]]\)=[[:alnum:]][[:alnum:].-_]*/\1/' env_name.yml > new_env_name.yml
```

Now you can create the conda env using new_env_name.yml
