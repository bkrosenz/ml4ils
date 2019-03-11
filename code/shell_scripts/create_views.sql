-- dont assume
drop view lg_lg;
drop view lg_wag;
drop view wag_lg;
drop view wag_wag;

create view lg_lg as 
       select i."1:1", i."1:2", i."1:3", i."1:4", i."2:2", i."2:3", i."2:4", i."3:3", i."3:4", i."4:4",
       i.vmr, i.length,
       g."1:1" as "g_1:1", g."1:2" as "g_1:2", g."1:3" as "g_1:3",
       g."1:4" as "g_1:4", g."2:2" as "g_2:2", g."2:3" as "g_2:3",
       g."2:4" as "g_2:4", g."3:3" as "g_3:3", g."3:4" as "g_3:4", g."4:4" as "g_4:4",
       s."1:1" as "s_1:1", s."1:2" as "s_1:2", s."1:3" as "s_1:3",
       s."1:4" as "s_1:4", s."2:2" as "s_2:2", s."2:3" as "s_2:3",
       s."2:4" as "s_2:4", s."3:3" as "s_3:3", s."3:4" as "s_3:4", s."4:4" as "s_4:4",
       g.vmr as g_vmr, g.length as g_length,
       s.topology as stop, g.topology as gtop, i.topology as itop, s.id as sid,
       s."1:2"-s."1:3" as ibl, s."1:1"-s."1:2" as ebl, s.length as s_length
from
       gene_trees as g
       inner join generated on g.id=generated.gid
       left join species_trees as s on s.id=generated.sid
       inner join
             (select inferred.true, gene_trees.* from inferred
                     inner join gene_trees on gene_trees.id=inferred.inf
                     where theta=0.01 and sim_model='LG' and infer_model='PROTCATLG'
                     ) as i
             on g.id=i.true;

--;;;;;;;;;;;;

create view wag_lg as 
       select i."1:1", i."1:2", i."1:3", i."1:4", i."2:2", i."2:3", i."2:4", i."3:3", i."3:4", i."4:4",
       i.vmr, i.length,
       g."1:1" as "g_1:1", g."1:2" as "g_1:2", g."1:3" as "g_1:3",
       g."1:4" as "g_1:4", g."2:2" as "g_2:2", g."2:3" as "g_2:3",
       g."2:4" as "g_2:4", g."3:3" as "g_3:3", g."3:4" as "g_3:4", g."4:4" as "g_4:4",
       s."1:1" as "s_1:1", s."1:2" as "s_1:2", s."1:3" as "s_1:3",
       s."1:4" as "s_1:4", s."2:2" as "s_2:2", s."2:3" as "s_2:3",
       s."2:4" as "s_2:4", s."3:3" as "s_3:3", s."3:4" as "s_3:4", s."4:4" as "s_4:4",
       g.vmr as g_vmr, g.length as g_length,
       s.topology as stop, g.topology as gtop, i.topology as itop, s.id as sid,
              s."1:2"-s."1:3" as ibl, s."1:1"-s."1:2" as ebl, s.length as s_length
from
       gene_trees as g
       inner join generated on g.id=generated.gid
       left join species_trees as s on s.id=generated.sid
       inner join
             (select inferred.true, gene_trees.* from inferred
                     inner join gene_trees on gene_trees.id=inferred.inf
                     where theta=0.01 and sim_model='WAG' and infer_model='PROTCATLG'
                     ) as i
             on g.id=i.true;

create view wag_wag as 
       select i."1:1", i."1:2", i."1:3", i."1:4", i."2:2", i."2:3", i."2:4", i."3:3", i."3:4", i."4:4",
       i.vmr, i.length,
       g."1:1" as "g_1:1", g."1:2" as "g_1:2", g."1:3" as "g_1:3",
       g."1:4" as "g_1:4", g."2:2" as "g_2:2", g."2:3" as "g_2:3",
       g."2:4" as "g_2:4", g."3:3" as "g_3:3", g."3:4" as "g_3:4", g."4:4" as "g_4:4",
       s."1:1" as "s_1:1", s."1:2" as "s_1:2", s."1:3" as "s_1:3",
       s."1:4" as "s_1:4", s."2:2" as "s_2:2", s."2:3" as "s_2:3",
       s."2:4" as "s_2:4", s."3:3" as "s_3:3", s."3:4" as "s_3:4", s."4:4" as "s_4:4",
       g.vmr as g_vmr, g.length as g_length,
       s.topology as stop, g.topology as gtop, i.topology as itop, s.id as sid,
              s."1:2"-s."1:3" as ibl, s."1:1"-s."1:2" as ebl, s.length as s_length
from
       gene_trees as g
       inner join generated on g.id=generated.gid
       left join species_trees as s on s.id=generated.sid
       inner join
             (select inferred.true, gene_trees.* from inferred
                     inner join gene_trees on gene_trees.id=inferred.inf
                     where theta=0.01 and  sim_model='WAG' and infer_model='PROTCATWAG'
                     ) as i
             on g.id=i.true;

create view lg_wag as 
       select i."1:1", i."1:2", i."1:3", i."1:4", i."2:2", i."2:3", i."2:4", i."3:3", i."3:4", i."4:4",
       i.vmr, i.length,
       g."1:1" as "g_1:1", g."1:2" as "g_1:2", g."1:3" as "g_1:3",
       g."1:4" as "g_1:4", g."2:2" as "g_2:2", g."2:3" as "g_2:3",
       g."2:4" as "g_2:4", g."3:3" as "g_3:3", g."3:4" as "g_3:4", g."4:4" as "g_4:4",
       s."1:1" as "s_1:1", s."1:2" as "s_1:2", s."1:3" as "s_1:3",
       s."1:4" as "s_1:4", s."2:2" as "s_2:2", s."2:3" as "s_2:3",
       s."2:4" as "s_2:4", s."3:3" as "s_3:3", s."3:4" as "s_3:4", s."4:4" as "s_4:4",
       g.vmr as g_vmr, g.length as g_length,
       s.topology as stop, g.topology as gtop, i.topology as itop, s.id as sid,
              s."1:2"-s."1:3" as ibl, s."1:1"-s."1:2" as ebl, s.length as s_length
from
       gene_trees as g
       inner join generated on g.id=generated.gid
       left join species_trees as s on s.id=generated.sid
       inner join
             (select inferred.true, gene_trees.* from inferred
                     inner join gene_trees on gene_trees.id=inferred.inf
                     where theta=0.01 and  sim_model='LG' and infer_model='PROTCATWAG'
                     ) as i
             on g.id=i.true;

-- select g."1:1" as g_1, g2."1:1" as i_1 from gene_trees as g join generated as gen on g.id=gen.gid join species_trees as s on s.id=gen.sid join inferred as i on i.true=g.id join gene_trees as g2 on g2.id=i.inf limit 5;
