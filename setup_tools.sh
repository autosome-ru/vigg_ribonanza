# script installs EternaFold, ViennaRNA, IPKnot and arnie packages in /tools directory
# assumes git is preinstalled
# assumes cmake version 3.23.1 is preinstalled
#!/bin/bash

mkdir -p tools
cd tools 
tools_path=$(pwd)

# download ViennaRNA package (version 2.6.4)
wget https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_6_x/ViennaRNA-2.6.4.tar.gz

tar -xvf ViennaRNA-2.6.4.tar.gz
rm ViennaRNA-2.6.4.tar.gz
mkdir ViennaRNA
cd ViennaRNA-2.6.4

# follow the steps specified in official tutorial https://github.com/ViennaRNA/ViennaRNA/blob/master/INSTALL
./configure --prefix=$tools_path/ViennaRNA
# this step takes a while
make
make check
make install
make installcheck

cd $tools_path

# installing EternaFold (version 1.3.1)
git clone https://github.com/eternagame/EternaFold.git
cd EternaFold/src
make

cd $tools_path

# installing IPknot (installation info: https://github.com/satoken/ipknot)
# set up GLPK solver (from https://www.gnu.org/software/glpk/) (version 5.0)
wget https://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz
tar -xvf glpk-5.0.tar.gz
rm glpk-5.0.tar.gz

# follow the install instructions in INSTALL file in glpk-5.0 directory
mkdir glpk
cd glpk-5.0
./configure --prefix=$tools_path/glpk
make
make check
make install

cd $tools_path

# installing IPknot from https://github.com/satoken/ipknot (version 1.1.0)
git clone https://github.com/satoken/ipknot.git
mv ipknot ipknot-1.1.0
cd ipknot-1.1.0
export PKG_CONFIG_PATH=$tools_path/ViennaRNA/lib/pkgconfig:$PKG_CONFIG_PATH
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..

# as glpk is installed in non-root directory the lines below modify CMakeCache.txt in order to specify correct paths
sed -i "s#GLPK_INCLUDE_DIR:PATH=.*#GLPK_INCLUDE_DIR:PATH=$tools_path/glpk/include#g" CMakeCache.txt
sed -i "s#GLPK_LIBRARY:FILEPATH=.*#GLPK_LIBRARY:FILEPATH=$tools_path/glpk/lib/libglpk.a#g" CMakeCache.txt
sed -i "s#GLPK_ROOT_DIR:PATH=.*#GLPK_ROOT_DIR:PATH=$tools_path/glpk#g" CMakeCache.txt
cmake --build .
cmake --install . --prefix $tools_path/ipknot


cd $tools_path
mkdir tmp
# installing arnie
git clone https://github.com/DasLab/arnie.git
cd arnie

# make arnie config file
echo -e "vienna_2: $tools_path/ViennaRNA-2.6.4/src/bin" > arnie_config.txt
echo -e "contrafold_2: None" >> arnie_config.txt
echo -e "rnastructure: None" >> arnie_config.txt
echo -e "rnasoft: None" >> arnie_config.txt
echo -e "vfold: None" >> arnie_config.txt
echo -e "nupack: None" >> arnie_config.txt
echo -e "eternafold: $tools_path/EternaFold/src" >> arnie_config.txt
echo -e "ipknot: $tools_path/ipknot/bin" >> arnie_config.txt
echo -e "TMP: $tools_path/tmp" >> arnie_config.txt





