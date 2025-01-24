import ase.db
import numpy as np
import argparse


description = 'Convert ase.db ito cfg format'

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-db', '--db_name', help='name of ase.db file to be converted', type=str, default="C-add-to-train.db")
parser.add_argument('-so', '--species_order', help='order of species in cfg file', type=str, default="Si O C Mo")
parser.add_argument('-sigma', '--use_stresses', help='whether of not to use stresses', type=bool, default=True)


args = parser.parse_args()
db = ase.db.connect(args.db_name)

cfg = args.db_name.strip("db")+"cfg"
cfg = open(cfg, "w")


specorder = args.species_order.split()
indices   = list(range(len(specorder)))


symbol2index = dict(zip(specorder, indices))

elem_map = ""
for s in specorder:
    elem_map += "%s = %i " %(s, symbol2index[s])
    

for row in db.select():
    try:
        row.energy
        cfg.write("BEGIN_CFG\n")
        cfg.write(" Size\n")
        cfg.write("  %i\n" %row.natoms)
        cfg.write(" Supercell\n")
        a, b, c = row.cell
        for e in [a, b, c]:
            cfg.write("  %2.5f\t %2.5f\t %2.5f\n" %(e[0], e[1], e[2]))

        cfg.write(" AtomData: id type cartes_x cartes_y cartes_z fx fy fz\n")

        symb   = row.symbols
        pos    = row.positions
        F      = row.forces
        
        try:
            en = row.free_energy

        except:
            en = row.energy
        v      = row.volume
        
        vir = [False]
        if args.use_stresses:
            try:
                sig    = row.stress
                vir    = -v*sig
            except:
                vir    = [False]


        for i in range(len(symb)):
            X = symbol2index[symb[i]]
            p = pos[i]
            f = F[i]
            cfg.write("\t\t %i\t %i\t %2.5f\t %2.5f\t %2.5f\t %2.5f\t %2.5f\t %2.5f\t\n" %(i+1, X, p[0], p[1], p[2], f[0], f[1], f[2]))
    
        cfg.write(" Energy\n")
        cfg.write("  %f \n" %en)

        if vir[0]:
            cfg.write(" PlusStress: xx yy zz yz xz xy\n")
            cfg.write("  %2.5f\t %2.5f\t %2.5f\t %2.5f\t %2.5f\t %2.5f\n" %(vir[0], vir[1], vir[2], vir[3], vir[4], vir[5]))   

        cfg.write("Feature\t Element order: %s\n" %elem_map)
        cfg.write("END_CFG\n\n")

    except:
        print("Structure %i not written to cfg" %row.id)

cfg.close()
