set terminal png
set output "gpu.png"
plot "kl" u 1:2 w l t "1024-ctxt", "kl2" u 1:2 pt 7 ps 2 t "2048-ctxt act. chkpt."
