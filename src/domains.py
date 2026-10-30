"""
EdgeInspect AI: Multi-Domain Industrial Defect Knowledge Base & Standards
Challenge: Arm Bharat AI-SoC Challenge 2026-27 (PS5: Industrial Defect Detection)

Covers all 6 challenge application domains:
1. Printed Circuit Board Inspection (IPC-A-610 / IPC-6012)
2. Steel & Metal Surface Defect Detection (ASTM A6 / ISO 9001 / NEU-DET)
3. Textile & Fabric Quality Control (ASTM D5430 4-Point System)
4. Packaging Integrity Verification (ISO 11607 / ASTM F88)
5. Automotive Component Inspection (IATF 16949 / VDA 6.3)
6. Pharmaceutical Tablet & Capsule Inspection (FDA cGMP / USP <1790>)
"""
from typing import Dict, Any, List

INDUSTRIAL_DOMAINS: Dict[str, Dict[str, Any]] = {
    "pcb": {
        "name": "Printed Circuit Board (PCB)",
        "icon": "🔌",
        "standard": "IPC-A-610 Class 2/3 & IPC-6012",
        "dataset_benchmark": "DeepPCB / PKU Benchmark",
        "classes": {
            "missing_hole": {
                "severity": "CRITICAL (CLASS 3 REJECT)",
                "root_cause": "Broken CNC carbide bit, spindle Z-axis misalignment, or Excellon drill file discrepancy.",
                "rework": "Halt drill spindle line. For unplated bare boards: secondary pass; if through-hole plating is complete: SCRAP.",
                "disposition": "REJECT / SCRAP"
            },
            "short": {
                "severity": "CRITICAL (CLASS 3 REJECT)",
                "root_cause": "Over-exposure of photoresist, incomplete copper chemical etching, or solder bridging on SMT pads.",
                "rework": "De-short under optical scope using micro-scalpel or precision solder wick with RMA flux.",
                "disposition": "REWORK REQUIRED"
            },
            "open_circuit": {
                "severity": "CRITICAL (CLASS 3 REJECT)",
                "root_cause": "Air bubble during etching, mechanical scratch tearing conductor line, or dust on optical phototool.",
                "rework": "Install IPC-7721 jumper wire (30 AWG Kynar) or micro-solder copper ribbon over discontinuity.",
                "disposition": "REWORK REQUIRED"
            },
            "mouse_bite": {
                "severity": "MAJOR DEFECT (CLASS 2/3 CONDITIONAL)",
                "root_cause": "Rough etching tank turbulence, localized dry film lift, or worn chemical nozzle.",
                "rework": "Measure remaining track conductor width. If remaining width > 80% per IPC-A-610: ACCEPT with touch-up conformal coating.",
                "disposition": "CONDITIONAL REWORK / ACCEPT"
            },
            "spur": {
                "severity": "MINOR DEFECT (CLASS 2 CONDITIONAL)",
                "root_cause": "Developing bath residual sludge, pinhole in dry-film photoresist, or chemical spray droplet deflection.",
                "rework": "Scrape spur with ceramic deburring tool ensuring dielectric distance > 0.15 mm to adjacent conductors.",
                "disposition": "REWORK / CLEAR"
            },
            "spurious_copper": {
                "severity": "MAJOR DEFECT (SHORT-CIRCUIT HAZARD)",
                "root_cause": "Undissolved copper flake during ferric chloride etch, localized resist contamination, or poor rinse agitation.",
                "rework": "Isolate with anti-static pick; verify clearance distance. Apply localized soldermask insulation.",
                "disposition": "REWORK REQUIRED"
            }
        }
    },

    "steel_metal": {
        "name": "Steel & Metal Surface",
        "icon": "🔩",
        "standard": "ASTM A6 / ISO 9001 / EN 10163",
        "dataset_benchmark": "NEU Surface Defect Database (NEU-DET)",
        "classes": {
            "rolled_in_scale": {
                "severity": "MAJOR DEFECT (STRUCTURAL / FATIGUE RISK)",
                "root_cause": "Iron oxide mill scale embedded into sheet surface during high-temperature rolling pass due to descaling nozzle pressure loss.",
                "rework": "Mechanical grinding within ASTM thickness allowance (max 5% depth reduction); acid pickling re-pass.",
                "disposition": "REWORK / GRINDING"
            },
            "patches": {
                "severity": "MAJOR SURFACE DEFECT",
                "root_cause": "Non-uniform cooling rate, localized roll wear, or slab surface decarburization in reheat furnace.",
                "rework": "Surface dressing with abrasive flapper wheel; surface roughness re-verification via profilometer.",
                "disposition": "REWORK REQUIRED"
            },
            "crazing": {
                "severity": "CRITICAL (STRESS CORROSION & FATIGUE CRACK)",
                "root_cause": "Copper or tin tramp element segregation along prior austenite grain boundaries during hot working (hot shortness).",
                "rework": "Perform ultrasonic crack depth inspection. Deep crazing (>0.3mm) requires immediate down-rating or coil scrap.",
                "disposition": "REJECT / SCRAP"
            },
            "pitted_surface": {
                "severity": "MODERATE (CORROSION SUSCEPTIBILITY)",
                "root_cause": "Acid over-pickling, moisture condensation during coil storage, or mechanical roll indentation.",
                "rework": "Degrease, apply rust inhibitor primer, and measure minimum wall thickness per structural tolerance.",
                "disposition": "ACCEPT WITH COATING / REWORK"
            },
            "inclusion": {
                "severity": "CRITICAL (INTERNAL DISCONTINUITY)",
                "root_cause": "Entrapped ladle slag (Al2O3 / SiO2) during continuous casting solidification in mold.",
                "rework": "Non-repairable structural defect. Divert sheet section to non-critical structural grade.",
                "disposition": "REJECT / DOWNGRADE"
            },
            "scratches": {
                "severity": "MINOR / COSMETIC",
                "root_cause": "Friction against misaligned runout table guides, roll slippage, or coil slitting shear contact.",
                "rework": "Buff surface with 240-grit silicon carbide paste if depth < 0.05 mm.",
                "disposition": "ACCEPT / BUFFING"
            }
        }
    },

    "textile_fabric": {
        "name": "Textile & Fabric Quality Control",
        "icon": "🧵",
        "standard": "ASTM D5430 (4-Point System) / ISO 8498",
        "dataset_benchmark": "AITEX / TILDA Fabric Defect Benchmark",
        "classes": {
            "hole_puncture": {
                "severity": "CRITICAL (4-POINT DEFECT - STRUCTURAL COMPROMISE)",
                "root_cause": "Needle breakage, broken heald wire on weaving loom, or excessive warp tension during beat-up.",
                "rework": "Cut out damaged roll section. Record defect coordinate on automated fabric inspection log.",
                "disposition": "CUT OUT / SPLICE"
            },
            "oil_stain": {
                "severity": "MAJOR (2 TO 4 POINTS)",
                "root_cause": "Loom camshaft oil seal leak, overhead lubricant dripping, or improper machine maintenance schedule.",
                "rework": "Spot-clean with eco-friendly ultrasonic solvent cleaner. Re-inspect under UV optical chamber.",
                "disposition": "REWORK / SCOURING"
            },
            "broken_yarn": {
                "severity": "MAJOR (WARP/WEFT DISCONTINUITY)",
                "root_cause": "Defective yarn tensile strength, knot failure, or abrasive yarn guide wear.",
                "rework": "Mending technician re-weaves manual repair stitch for high-end fabrics, or grade B downgrading.",
                "disposition": "MEND / DOWNGRADE"
            },
            "color_shading": {
                "severity": "MODERATE (DYEING NON-UNIFORMITY)",
                "root_cause": "Temperature gradient across stenter frame or uneven padding liquor pick-up in dye bath.",
                "rework": "Re-batch for stripping and level re-dyeing if lot size justifies energy cost.",
                "disposition": "RE-DYE / DOWNGRADE"
            }
        }
    },

    "packaging": {
        "name": "Packaging Integrity Verification",
        "icon": "📦",
        "standard": "ISO 11607 (Sterile Barrier) / ASTM F88 / ASTM F1929",
        "dataset_benchmark": "MVTec AD (Bottle, Capsule, Cable, Box)",
        "classes": {
            "seal_breach": {
                "severity": "CRITICAL (LOSS OF STERILITY / BARRIER INTEGRITY)",
                "root_cause": "Sealing bar temperature drop, pressure loss on pneumatic cylinder, or product entrapment in seal jaw.",
                "rework": "Reject package to dump bin. Check thermocouple PID parameters on sealing jaw.",
                "disposition": "REJECT / RE-PACK"
            },
            "label_misalignment": {
                "severity": "MINOR / COMPLIANCE VIOLATION",
                "root_cause": "Dispenser roll friction fluctuation, vacuum belt slippage, or photo-eye trigger timing jitter.",
                "rework": "Peel and re-apply label if backing adhesive permits, or adjust applicator servo delay offset.",
                "disposition": "RE-LABEL / REWORK"
            },
            "container_dent": {
                "severity": "MODERATE (AESTHETIC / STRUCTURAL BUCKLING)",
                "root_cause": "Conveyor accumulation line pressure surge or pneumatic ejector impact strike.",
                "rework": "Sort unit; inspect for micro-cracks or pinholes. Reject if internal liner is compromised.",
                "disposition": "REJECT / RECYCLE"
            },
            "contamination_foreign_object": {
                "severity": "CRITICAL (CONTAMINATION HAZARD)",
                "root_cause": "Air handling unit filter bypass, packaging hopper particulate fallout, or static charge attracting dust.",
                "rework": "Quarantine upstream batch. Clean HEPA blow-off ionizer bar.",
                "disposition": "QUARANTINE BATCH"
            }
        }
    },

    "automotive": {
        "name": "Automotive Component Inspection",
        "icon": "🚗",
        "standard": "IATF 16949 / VDA 6.3 / ISO 9001",
        "dataset_benchmark": "KolektorSDD / MVTec Surface Inspection",
        "classes": {
            "casting_porosity": {
                "severity": "CRITICAL (PRESSURE VESSEL & HYDROSTATIC FAILURE)",
                "root_cause": "Hydrogen gas entrapment during die-cast pour or inadequate runner/vent overflow gating.",
                "rework": "Vacuum impregnation resin sealing if pore diameter < 0.25 mm; otherwise SCRAP component.",
                "disposition": "IMPREGNATION / SCRAP"
            },
            "crack_fracture": {
                "severity": "SAFETY CRITICAL (ZERO DEFECT MANDATE)",
                "root_cause": "Thermal quench shock, residual machining stress, or excessive clamping hydraulic pressure.",
                "rework": "Zero rework allowed per automotive safety standards. Immediate scrap & tool audit.",
                "disposition": "SCRAP (SAFETY CRITICAL)"
            },
            "burr_flash": {
                "severity": "MODERATE (ASSEMBLY CLEARANCE INTERFERENCE)",
                "root_cause": "Worn parting line on forging die or dull CNC endmill cutting tool.",
                "rework": "Automated thermal deburring or manual precision chamfering with pneumatic pencil grinder.",
                "disposition": "DEBURR / REWORK"
            },
            "weld_spatter": {
                "severity": "MINOR / FUNCTIONAL INTERFERENCE",
                "root_cause": "Incorrect shielding gas flow rate, excessive MIG welding voltage, or contaminated welding wire.",
                "rework": "Chop and wire-brush spatter; verify mating gasket seal face roughness Ra < 3.2 um.",
                "disposition": "CLEAN / ACCEPT"
            }
        }
    },

    "pharma": {
        "name": "Pharmaceutical Tablet & Capsule",
        "icon": "💊",
        "standard": "FDA 21 CFR Part 211 (cGMP) / USP <1790> / EP 2.9.20",
        "dataset_benchmark": "MVTec Pill / Tablet Defect Benchmark",
        "classes": {
            "chipped_edge": {
                "severity": "CRITICAL (DOSAGE UNIFORMITY FAILURE)",
                "root_cause": "Excessive tablet press ejection punch force, high friability, or worn die bore bevel.",
                "rework": "Divert tablet to waste gate. Adjust rotary press feeder speed and main compression force.",
                "disposition": "REJECT / DISCARD"
            },
            "capping_lamination": {
                "severity": "CRITICAL (STRUCTURAL INTEGRITY LOSS)",
                "root_cause": "Air entrapment during high-speed compression, excessive fines in granulation, or low moisture.",
                "rework": "Halt press immediately. De-aerate granulation hopper; adjust pre-compression roller ratio.",
                "disposition": "REJECT / LINE AUDIT"
            },
            "mottling_discoloration": {
                "severity": "MAJOR (CONTAMINATION / DEGRADATION RISK)",
                "root_cause": "Active pharmaceutical ingredient (API) thermal degradation, pigment migration, or lubricant contamination.",
                "rework": "Quarantine lot. Perform HPLC assay testing for chemical degradation products.",
                "disposition": "QUARANTINE / LAB TEST"
            },
            "capsule_dent_split": {
                "severity": "CRITICAL (FILL LEAKAGE HAZARD)",
                "root_cause": "Misaligned capsule closing station pins, brittle gelatin/HPMC shell from low humidity storage.",
                "rework": "Eject damaged shell. Vacuum recover spilled powder; calibrate closing pin stroke depth.",
                "disposition": "REJECT / RE-FILL"
            }
        }
    }
}
