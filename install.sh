#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "\nAfter registering at https://arteq.is.tue.mpg.de/, provide your credentials:"
read -p "Username:" username
read -s -p "Password: " password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading ArtEq..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=arteq&sfile=data.zip&resume=1' -O 'data.zip' --no-check-certificate --continue
unzip data.zip -d data/
rm data.zip

conda env create -f environment.yml
