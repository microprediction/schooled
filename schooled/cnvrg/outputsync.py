from schooled.wherami import RUNNING_LOCALLY

if not RUNNING_LOCALLY:

    import os
    from cnvrgv2 import Project, Experiment

    def sync_previous(flow_name):
        proj = Project()
        flow = proj.flows.get(flow_name)

        # Latest flow version
        flow_version = next(flow.flow_versions.list())
        # Experiments in the Latest run of the Flow
        flow_tasks = flow_version.info()["fv_status"]["task_statuses"]
        # Experiment/s in the previous task
        prev_task_experiments = flow_tasks[-2]["experiments"]

        # save current path/working directory
        cwd = os.getcwd()

        # create folders with experiments names and pull artifacts to then - inside "WORKING_DIR/output/" -
        for exp in prev_task_experiments:
            e = Experiment(slug=exp["id"])  # get experiment by slug
            exp_output_path = "output/" + e.title

            os.system("mkdir -p " + exp_output_path.replace(" ", "\ "))
            os.chdir(cwd + "/" + exp_output_path)
            e.pull_artifacts()
            os.chdir(cwd)



else:
    def sync_previous():
        raise NotImplementedError('sync is being called from local - oops')
