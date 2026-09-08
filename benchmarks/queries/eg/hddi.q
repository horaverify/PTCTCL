E[] ST1.z <= 40
E[] ST1.z <= 40 && !ST2.station_z_async
E[] (ST1.station_z_idle || ST1.station_z_sync) && ST1.z <= 40
E[] !(ST1.station_z_sync && ST2.station_z_sync) && ST1.z <= 40
E[] ST1.z <= 20 && (ST2.z <= 15 || ST2.z >= 18)
