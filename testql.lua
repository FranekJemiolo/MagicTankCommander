function testQL()
    -- Fceux path, not using opengl, loading savestate 1, 
    -- loading main lua script, rom location
    os.execute("/home/franek/JNP3/fceux-2.2.2/bin/fceux --opengl 0" ..
        "--loadstate 1.fcs --loadlua /home/franek/JNP3/model.lua " .. 
        "/home/franek/JNP3/roms/batlcity.nes")
end

testQL()