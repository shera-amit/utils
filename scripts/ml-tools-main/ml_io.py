from ase import Atoms,Atom
from ase.io import read,write
import numpy as np
import pandas as pd



def write_mlip(filename,atoms,specorder,energy_name="free_energy",force_name="forces",virial_name="virial",energy_correction=None):
	outfile = open(filename,"w")
	if type(atoms) != list:
		write_mlip_single(outfile,atoms,specorder,energy_name=energy_name,force_name=force_name,virial_name=virial_name,energy_correction=energy_correction)
	else:
		for i in range(0,len(atoms)):
			write_mlip_single(outfile,atoms[i],specorder,energy_name=energy_name,force_name=force_name,virial_name=virial_name,energy_correction=energy_correction)
	outfile.close()
	
	
def write_mlip_single(outfile,atoms,specorder,energy_name="free_energy",force_name="forces",virial_name="virial",energy_correction=None):
	file_string = ""
	file_string += "BEGIN_CFG\n"
	file_string += " Size\n"
	file_string += "    "+str(len(atoms))+"\n"
	file_string += " Supercell\n"
	cell = atoms.get_cell(complete=True)
	file_string += "         "+str(f"{cell[0,0]:.6f}")+"    "+str(f"{cell[0,1]:.6f}")+"    "+str(f"{cell[0,2]:.6f}")+"\n"
	file_string += "         "+str(f"{cell[1,0]:.6f}")+"    "+str(f"{cell[1,1]:.6f}")+"    "+str(f"{cell[1,2]:.6f}")+"\n"
	file_string += "         "+str(f"{cell[2,0]:.6f}")+"    "+str(f"{cell[2,1]:.6f}")+"    "+str(f"{cell[2,2]:.6f}")+"\n"
	if force_name!=None:
		file_string += " AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz\n"
		if force_name=="forces":
			try:
				forces = atoms.arrays["forces"]
			except:
				forces = atoms.get_forces()
		else:
			forces = atoms.arrays[force_name]
	else:
		file_string += " AtomData:  id type       cartes_x      cartes_y      cartes_z\n"
	id = 1
	for atom in atoms:
		type = specorder.index(atom.symbol)
		if force_name!=None:
			x,y,z = atom.position
			fx, fy, fz = forces[id-1]
			if x<0:
				disx="      "
			else:
				disx="       "
			if y<0:
				disy="     "
			else:
				disy="      "
			if z<0:
				disz="     "
			else:
				disz="      "
			if fx<0:
				disfx="    "
			else:
				disfx="     "
			if fy<0:
				disfy="   "
			else:
				disfy="    "
			if fz<0:
				disfz="   "
			else:
				disfz="    "
			file_string += "             "+str(id)+"    "+str(type)+disx+str(f"{x:.6f}")+disy+str(f"{y:.6f}")+disz+str(f"{z:.6f}")+disfx+str(f"{fx:.6f}")+disfy+str(f"{fy:.6f}")+disfz+str(f"{fz:.6f}")+"\n"
		else:
			x,y,z = atom.position
			if x<0:
				disx="      "
			else:
				disx="       "
			if y<0:
				disy="      "
			else:
				disy="       "
			if z<0:
				disz="      "
			else:
				disz="       "
			file_string += "             "+str(id)+"    "+str(type)+disx+str(f"{x:.6f}")+disy+str(f"{y:.6f}")+disz+str(f"{z:.6f}")+"\n"
		id = id+1
	if energy_name!=None:
		file_string += " Energy\n"
		if energy_name=="free_energy":
			try:
				energy = atoms.info["free_energy"]
			except:
				energy = atoms.get_potential_energy(force_consistent=True)
		elif energy_name=="energy":
			try:
				energy = atoms.info["energy"]
			except:
				energy = atoms.get_potential_energy()
		else:
			energy = atoms.info[energy_name]
		if energy_correction!=None:
			symbols = atoms.get_chemical_symbols()
			correction = 0
			for i in range(0,len(specorder)):
				num = symbols.count(specorder[i])
				correction = correction-num*energy_correction[i]
		else:
			correction = 0
		new_energy = energy+correction
		file_string += "        "+str(f"{new_energy:.12f}")+"\n"
	if virial_name!=None:
		file_string += " PlusStress: xx    yy    zz    yz    xz    xy\n"
		virial = atoms.info[virial_name]
		new_virial = [virial[0,0],virial[1,1],virial[2,2],virial[1,2],virial[0,2],virial[0,1]]
		file_string += "        "+str(f"{new_virial[0]:.5f}")+"    "+str(f"{new_virial[1]:.5f}")+"    "+str(f"{new_virial[2]:.5f}")+"    "+str(f"{new_virial[3]:.5f}")+"    "+str(f"{new_virial[4]:.5f}")+"    "+str(f"{new_virial[5]:.5f}")+"\n"
	for k in range(0,len(specorder)):
		file_string += " Feature ID: "+str(k)+" = "+specorder[k]+"\n"
	if energy_correction!=None and energy_name!=None:
		file_string += " Feature Uncorrected_Energy = "+str(energy)+"\n"
	file_string +="END_CFG\n"
	outfile.write(file_string)
	
	
def write_pacemaker(filename,atoms,specorder=None,energy_correction=None,energy_name=None,force_name=None):
	data = {"ase_atoms": [],
		"forces"   : [],
		"energy_corrected": [],
		"energy" : []}
	if type(atoms)==list:
		for a in atoms:
			if force_name=="forces":
				try:
					forces = a.arrays["forces"]
				except:
					forces = a.get_forces()
			else:
				if force_name==None:
					forces = a.get_forces()
				else:
					forces = a.arrays[force_name]

			if energy_name=="energy":
				try:
					energy = a.info["energy"]
				except:
					energy = a.get_potential_energy()
			elif energy_name=="free_energy":
				try:
					energy = a.info["free_energy"]
				except:
					energy = a.get_potential_energy(force_consistent=True)
			else:
				if energy_name==None:
					energy = a.get_potential_energy(force_consistent=True)
				else:
					energy = a.info[energy_name]

			data["forces"].append(forces)
			data["energy"].append(energy)
			data["ase_atoms"].append(a)
			symbols = a.get_chemical_symbols()
			if energy_correction==None:
				data["energy_corrected"].append(energy)
			else:
				correction = 0
				for i in range(0,len(specorder)):
					num = symbols.count(specorder[i])
					correction = correction-num*energy_correction[i]
				data["energy_corrected"].append(energy+correction)
	else:
		if force_name=="forces":
				try:
					forces = atoms.arrays["forces"]
				except:
					forces = atoms.get_forces()
		else:
			if force_name==None:
				forces = atoms.get_forces()
			else:
				forces = atoms.arrays[force_name]

		if energy_name=="energy":
			try:
				energy = atoms.info["energy"]
			except:
				energy = atoms.get_potential_energy()
		elif energy_name=="free_energy":
			try:
				energy = atoms.info["free_energy"]
			except:
				energy = atoms.get_potential_energy(force_consistent=True)
		else:
			if energy_name==None:
				energy = atoms.get_potential_energy(force_consistent=True)
			else:
				energy = atoms.info[energy_name]
		data["forces"].append(forces)
		data["energy"].append(energy)
		data["ase_atoms"].append(atoms)
		symbols = atoms.get_chemical_symbols()
		if energy_correction==None:
			data["energy_corrected"].append(energy)
		else:
			correction = 0
			for i in range(0,len(specorder)):
				num = symbols.count(specorder[i])
				correction = correction-num*energy_correction[i]
			data["energy_corrected"].append(energy+correction)
	df = pd.DataFrame(data)
	df.to_pickle(filename,compression="gzip",protocol=4)

def write_nnp(filename,atoms,specorder=None,energy_correction=None,energy_name=None,force_name=None,charge_name=None):
	outfile = open(filename,"w")
	if type(atoms)==list:
		for i in range(0,len(atoms)):
			write_nnp_single(outfile,atoms[i],specorder=specorder,energy_correction=energy_correction,energy_name=energy_name,force_name=force_name,charge_name=charge_name)
	else:
		write_nnp_single(outfile,atoms,specorder=None,energy_correction=energy_correction,energy_name=energy_name,force_name=force_name,charge_name=charge_name)
	outfile.close()
		
			

def write_nnp_single(outfile,atoms,specorder=None,energy_correction=None,energy_name=None,force_name=None,charge_name=None):
	file_string = ""
	file_string += "begin\n"
	if "config_type" in atoms.info:
		file_string += "comment "+str(atoms.info["config_type"])+"\n"
	cell = atoms.get_cell(complete=True)
	file_string += "lattice  "+str(cell[0,0]/0.529177208)+"    "+str(cell[0,1]/0.529177208)+"    "+str(cell[0,2]/0.529177208)+"\n"
	file_string += "lattice  "+str(cell[1,0]/0.529177208)+"    "+str(cell[1,1]/0.529177208)+"    "+str(cell[1,2]/0.529177208)+"\n"
	file_string += "lattice  "+str(cell[2,0]/0.529177208)+"    "+str(cell[2,1]/0.529177208)+"    "+str(cell[2,2]/0.529177208)+"\n"
	id = 1
	if force_name == "forces":
		if force_name in atoms.arrays:
			forces = atoms.arrays[force_name]
		else:
			forces = atoms.get_forces()
	else:
		if force_name != None:
			forces = atoms.arrays[force_name]
		
	if charge_name in atoms.arrays:
		charges = atoms.arrays[charge_name]
		q = True
	else:
		q = False
	for atom in atoms:
		x,y,z = atom.position/0.529177208
		if force_name != None:
			fx, fy, fz = forces[id-1]/51.42208619083232
		sym = atom.symbol
		if q:
			ch = charges[id-1]
			if force_name != None:
				file_string += "atom "+str(x)+" "+str(y)+" "+str(z)+" "+sym+" "+str(ch)+" 0 "+str(fx)+" "+str(fy)+" "+str(fz)+"\n"
			else:
				file_string += "atom "+str(x)+" "+str(y)+" "+str(z)+" "+sym+" "+str(ch)+" 0 "+str(0)+" "+str(0)+" "+str(0)+"\n"
		else:
			ch = 0
			if force_name != None:
				file_string += "atom "+str(x)+" "+str(y)+" "+str(z)+" "+sym+" "+str(ch)+" 0 "+str(fx)+" "+str(fy)+" "+str(fz)+"\n"
			else:
				file_string += "atom "+str(x)+" "+str(y)+" "+str(z)+" "+sym+" "+str(ch)+" 0 "+str(0)+" "+str(0)+" "+str(0)+"\n"
		id = id+1
	
	if energy_name == "free_energy":
		if "free_energy" in atoms.info:
			energy = atoms.info["free_energy"]
		else:
			energy = atoms.get_potential_energy(force_consistent=True)
	elif energy_name=="energy":
		if "energy" in atoms.info:
			energy = atoms.info["energy"]
		else:
			energy = atoms.get_potential_energy()
	else:
		if energy_name != None:
			energy = atoms.info[energy_name]
		
	symbols = atoms.get_chemical_symbols()
	if energy_correction!=None:
		correction = 0
		for i in range(0,len(specorder)):
			num = symbols.count(specorder[i])
			correction = correction-num*energy_correction[i]
	else:
		correction = 0
	if energy_name != None:
		new_energy = energy+correction
		file_string += "energy "+str(new_energy/27.211396)+"\n"
	else:
		file_string += "energy "+str(0)+"\n"
	if q:
		file_string += "charge "+str(np.round(np.sum(charges),2))+"\n"
	else:
		file_string += "charge 0.00000\n"
	file_string += "end\n"
	outfile.write(file_string)


def write_ml(filename,atoms,format,energy_correction=None,specorder=None,energy_name="free_energy",force_name="forces",virial_name="virials",charge_name="charge"):
	if format=="mlip":
		write_mlip(filename,atoms,specorder,energy_correction=energy_correction,energy_name=energy_name,force_name=force_name,virial_name=virial_name)
	elif format=="pacemaker":
		write_pacemaker(filename,atoms,energy_correction=energy_correction,energy_name=energy_name,force_name=force_name,specorder=specorder)
	elif format=="nnp":
		write_nnp(filename,atoms,specorder=specorder,energy_correction=energy_correction,energy_name=energy_name,force_name=force_name,charge_name=charge_name)



def read_nnp(filename):
	infile = open(filename,"r")
	data   = infile.readlines()
	atoms_list = []
	count = 0
	for line in data:
		if "end" in line:
			atoms.set_cell(np.array([a1,a2,a3]))
			atoms.set_pbc(True)
			atoms.arrays["forces"]= np.array(force_list)
			atoms_list.append(atoms.copy())
		if "begin" in line:
			atoms = Atoms()
			force_list = []
			lat_count = 0
		if "comment" in line: 
			atoms.info["runner_comment"]=line.split()[1:]
		if "lattice" in line:
			x1 = float(line.split()[1])*0.529177208
			x2 = float(line.split()[2])*0.529177208
			x3 = float(line.split()[3])*0.529177208
			if lat_count==0:
				a1 = np.array([x1,x2,x3])
			elif lat_count==1:
				a2 = np.array([x1,x2,x3])
			elif lat_count==2:
				a3 = np.array([x1,x2,x3])
			lat_count =lat_count+1
		if "atom" in line:
			x1 = float(line.split()[1])*0.529177208
			x2 = float(line.split()[2])*0.529177208
			x3 = float(line.split()[3])*0.529177208
			f1 = float(line.split()[7])*51.42208619083232
			f2 = float(line.split()[8])*51.42208619083232
			f3 = float(line.split()[9])*51.42208619083232
			force_list.append([f1,f2,f3])
			symbol = line.split()[4]
			atom = Atom(symbol, position=[x1,x2,x3])
			atoms.append(atom)
		if "energy" in line:
			e = float(line.split()[1])
			atoms.info["energy"]=e*27.211396
		count = count+1
	return atoms_list


def read_mlip(filename,specorder):
	infile = open(filename,"r")
	data = infile.readlines()
	count = 0
	atoms_list = []
	while count < len(data):
		if "BEGIN_CFG" in data[count]:
			atoms = Atoms()
			force_array = []
		if "Size" in data[count]:
			num = int(data[count+1])
			count+=1
		if "Supercell" in data[count]:
			x,y,z = data[count+1].split()
			cella = np.array([float(x),float(y),float(z)])
			x,y,z = data[count+2].split()
			cellb = np.array([float(x),float(y),float(z)])
			x,y,z = data[count+3].split()
			cellc = np.array([float(x),float(y),float(z)])
			atoms.set_cell(np.array([cella,cellb,cellc]))
			count += 1
		if "Energy" in data[count]:
			atoms.info["energy"]=float(data[count+1])
			count += 1
		if "PlusStress" in data[count]:
			xx,yy,zz,yz,xz,xy = data[count+1].split()
			atoms.info["virial"]= np.array([[float(xx),float(xy),float(xz)],[float(xy),float(yy),float(yz)],[float(xz),float(yz),float(zz)]])
			atoms.info["stress"]= -np.array([[float(xx),float(xy),float(xz)],[float(xy),float(yy),float(yz)],[float(xz),float(yz),float(zz)]])/atoms.get_volume()
			count += 1
		if "AtomData" in data[count]:
			for i in range(0,num):
				line = data[count+1+i]
				id,typ,x,y,z,fx,fy,fz = line.split()
				if specorder==None:
					atom = Atom(symbol=int(typ)+1,position=[x,y,z])
				else:
					atom = Atom(symbol=specorder[int(typ)],position=[x,y,z])
				force_array.append([float(fx),float(fy),float(fz)])
				atoms.append(atom)
			count=count+num
		if "END_CFG" in data[count]:
			atoms.arrays["forces"] = np.array(force_array)
			atoms_list.append(atoms)
		count+=1
	return atoms_list



def read_runner_output(path="."):
	infile = open(path+"/energy.out","r")
	data = infile.readlines()
	energy = float(data[1].split()[3])/27.211396
	infile.close()
	forces = np.loadtxt(path+"/nnforces.out",skiprows=1,usecols=(5,6,7))*51.42208619083232
	stress = np.loadtxt(path+"/nnstress.out",skiprows=1,usecols=(1,2,3))*27.211396/(0.529177208**3)
	return energy, forces, stress



def read_ml(filename,format,specorder=None):
	if format=="nnp":
		return read_nnp(filename)
	elif format=="mlip":
		return read_mlip(filename,specorder=specorder)
		

