import jsbsim

fdm = jsbsim.FGFDMExec(root_dir="/home/gordon/Apps/jsbsim-code")
fdm.load_model('c172x')
fdm.load_ic('reset00', useStoredPath=True)
fdm.run_ic()

prop = fdm['ic/vc-kts']
prop2 = fdm['qwerty']
prop3 = fdm['ic/vc-ktsRW']

testy1 = fdm.set_property_value('uiop', 42.0)
testy2 = fdm.get_property_value('uiop')

catalog = {}
for item in fdm.query_property_catalog('ic'):
    prop_name = item.split(" ")[0]
    catalog[item] = fdm.get_property_value(prop_name)

