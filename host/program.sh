service lightdm stop
rmmod -f altvipfb
rmmod cfbfillrect
rmmod cfbimgblt
rmmod cfbcopyarea
aocl program /dev/acl0 pipeline.aocx
modprobe altvipfb
service lightdm start

