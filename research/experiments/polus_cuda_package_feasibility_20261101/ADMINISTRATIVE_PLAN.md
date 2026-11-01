# Conditional isolated installation and driver rollback plan

**NOT EXECUTED. Current disposition: no approved safe production upgrade path
established for POWER8/P100/RHEL7.5.** The feasibility task authorizes metadata
queries and dependency dry-runs, not installation or system maintenance.

1. Obtain administrator/vendor confirmation for the exact machine generation,
   P100 SXM2/NVLink firmware, OS/minor release, per-node running kernel, matching
   headers/toolchain, driver flavor and support/security lifecycle. Official
   CUDA11.8 POWER qualification is POWER9/RHEL8. Do not reinterpret x86 RHEL7
   support as POWER8 support, install EL8 RPMs on EL7, use `--override`/force
   package flags, patch proprietary binaries, or substitute open kernel modules
   for Pascal. If qualification fails, choose an approved supported GPU host;
   no shared-node upgrade should proceed on this report alone.
2. Administrators must separately authorize and schedule a **single canary
   compute node**, never an unannounced change on the shared frontend. Drain it
   through LSF, wait for other users' jobs to finish naturally, and coordinate
   CPU/GPU reservations, filesystem/network services and a maintenance window.
   Do not cancel jobs. The research account's normal-queue ACL is not permission
   to administer nodes; gpulong's ACL remains incompatible. Verify per-job
   one-GPU visibility/reservation after any approved change. No LSF/admin state
   was changed by this investigation.
3. Before an approved transaction, make a recoverable node image/snapshot and
   verify out-of-band console and boot recovery. Archive the exact current
   R440.33.01 RPM set, dependencies/signatures/hashes, RPM database/list,
   repository configuration, DKMS source/build state, original kernel/devel/
   headers, bootloader/initramfs, modprobe/udev rules, library links and service
   settings. Keep the current kernel boot option. Confirm the backups can
   restore the complete module/user-library tuple; filesystem copies of one
   `libcuda.so` do not constitute driver rollback.
4. Preserve CUDA10.2 and every existing application activation path. If a
   supported newer driver transaction is approved, prefer the site's existing
   RPM/DKMS administration mechanism; do not mix a runfile over RPM-managed
   files. Administrators must resolve the entire distribution-specific package
   plan and matching kernel integration in staging. No module unload/build,
   installer or reboot command is supplied here as ready-to-run.
5. Install the new **user environment only after separate authorization** into
   a newly named research prefix. PROPOSED_TECHNICAL_LOCK.json records a solved
   alternative: defaults Python3.10.13, pinned RocketCE1.9.1 CUDA Torch2.0.1,
   CUDA11.8/cuDNN8.8/NCCL2.17, metrics libraries and optional scatter/sparse;
   universal PyG2.4 wheel separately pinned. First audit the full binary ISA/ABI
   closure. Exclude stock RocketCE Python with POWER9 flags. A solved plan may
   contain other dependencies requiring POWER9; none is allowed to run until
   reviewed. Retain package receipts and verify each downloaded hash. Do not
   rewrite metadata, strip dependencies or use `--no-deps` to force a failing
   Conda solve. The pip no-deps dry-run in this report only checked a pure wheel
   whose named core dependencies were independently included in Conda.
6. Health gate on the isolated node: matching driver module/user libraries,
   two healthy P100s, no new driver faults, healthy NVLink/device nodes and LSF
   inventory/reservation, existing CUDA10.2 users' representative workloads,
   storage/network/OFED integration where applicable, and unchanged CPU
   environment/activation hashes and existing CPU tests. New driver backward
   compatibility is a reason to test old workloads, not assume equality.
   Stop if any platform/prerequisite check rejects the configuration.
7. Only then, under a separately authorized bounded LSF technical job, verify
   native imports/arch list/CUDA allocation, CUDA PyG aggregation and gradients,
   GCN/SAGE structural surgery and reload, and the unchanged algorithms on
   frozen inputs. Use isolated per-device caches, original validation split,
   masked test labels, synchronized timing and allocated/reserved memory.
   Record all output/action/Q numerical differences. No training or full56
   campaign is part of this plan, and no parallel/multi-GPU Q is proposed.
8. **Rollback triggers:** failed driver/platform validation, module/library
   mismatch, GPU disappearance/faults, failed old-workload or CPU regression,
   broken LSF GPU allocation, or an administrator decision to abort. Keep the
   canary drained; restore the original node image or complete R440 RPM/DKMS/
   kernel/library/configuration set together; reboot via the approved recovery
   procedure; verify original driver, two GPUs, CUDA10.2 workloads, CPU prefix
   and scheduler health before returning it to service. Restore from a full
   image if package rollback cannot be validated. Never test rollback on busy
   production nodes. New user-prefix rollback is deactivation/quarantine of
   that new prefix; the working CPU prefix must never have been edited.

This report establishes neither runtime compatibility nor a maintenance approval.
The safe immediate action is to keep the running cluster unchanged while the
administrator resolves platform qualification, with the inspected package and
solver manifests available for review.
