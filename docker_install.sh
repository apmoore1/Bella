# Taken from:
# https://docs.docker.com/v17.09/engine/installation/linux/docker-ce/ubuntu/#install-using-the-repository
sudo apt-get remove docker docker-engine docker.io
sudo apt-get -y update
sudo apt-get -y install apt-transport-https \
                        ca-certificates \
                        curl \
                        software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
     "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) \
     stable"
sudo apt-get -y update
sudo apt-get -y install docker-ce=17.12.0~ce-0~ubuntu
