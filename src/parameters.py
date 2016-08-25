#########################################################################
# This code reads the input parameters the code uses the function and
# method described in:
# https://wiki.python.org/moin/ConfigParserExamples
#########################################################################
import ConfigParser
import sys

input_param = sys.argv[1]

Config = ConfigParser.ConfigParser()
Config.read("../params/" + input_param)

def ConfigSectionMap(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1

# factor to convert from km/s to kpc/Gyr
conv_factor = 1.0227121650537077

path = str(ConfigSectionMap("params")['path'])
filename = str(ConfigSectionMap("params")['filename'])
x_sat = float(ConfigSectionMap("params")['xsat']) # 
y_sat = float(ConfigSectionMap("params")['ysat']) # 
z_sat = float(ConfigSectionMap("params")['zsat']) # 
vx_sat = float(ConfigSectionMap("params")['vxsat']) * conv_factor # 
vy_sat = float(ConfigSectionMap("params")['vysat']) * conv_factor # 
vz_sat = float(ConfigSectionMap("params")['vzsat']) * conv_factor # 
M_sat = float(ConfigSectionMap("params")['msat']) * 1E10 # 
Host_model = float(ConfigSectionMap("params")['hostmodel']) # 
c_sat = float(ConfigSectionMap("params")['csat']) # 
Sat_model = float(ConfigSectionMap("params")['satmodel']) # 
Rvir_sat = float(ConfigSectionMap("params")['rvirsat']) # 
rs_sat = float(ConfigSectionMap("params")['rssat']) # 
x_host = float(ConfigSectionMap("params")['xhost']) # 
y_host = float(ConfigSectionMap("params")['yhost']) # 
z_host = float(ConfigSectionMap("params")['zhost']) # 
vx_host = float(ConfigSectionMap("params")['vxhost']) * conv_factor # 
vy_host = float(ConfigSectionMap("params")['vyhost']) * conv_factor # 
vz_host = float(ConfigSectionMap("params")['vzhost']) * conv_factor # 
M_host = float(ConfigSectionMap("params")['mhost']) * 1E10 # 
Rvir_host =  float(ConfigSectionMap("params")['rvirhost']) # 
rs_host = float(ConfigSectionMap("params")['rshost']) # 
c_host = float(ConfigSectionMap("params")['chost']) # 
M_disk = float(ConfigSectionMap("params")['mdisk']) * 1E10 
a_disk = float(ConfigSectionMap("params")['adisk'])
b_disk = float(ConfigSectionMap("params")['bdisk'])
M_bulge = float(ConfigSectionMap("params")['mbulge']) * 1E10
rh = float(ConfigSectionMap("params")['rhdisk'])
Host_move = float(ConfigSectionMap("params")['hostmove'])
cl = float(ConfigSectionMap("params")['coulombl'])
alpha_df_sat = float(ConfigSectionMap("params")['alphasat'])
alpha_df_host = float(ConfigSectionMap("params")['alphahost'])
L = float(ConfigSectionMap("params")['lvalue'])
C = float(ConfigSectionMap("params")['cvalue'])
Host_df = float(ConfigSectionMap("params")['hostdf'])
ac = float(ConfigSectionMap("params")['ac'])
contra_output = str(ConfigSectionMap("params")['contraout'])
time = float(ConfigSectionMap("params")['time']) # 
direction = float(ConfigSectionMap("params")['direction'])

out_param = open(path + input_param + "-savedvalues", "w")
out_param.write("xsat%f \n"%(x_sat))
out_param.write("ysat%f \n"%(y_sat))
out_param.write("zsat%f \n"%(z_sat))
out_param.write("vxsat%f \n"%(vx_sat))
out_param.write("vysat%f \n"%(vy_sat))
out_param.write("vzsat%f \n"%(vz_sat))
out_param.write("Msat%f \n"%(M_sat))
out_param.close()
