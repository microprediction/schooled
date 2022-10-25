from schooled.wherami import RUNNING_LOCALLY

if not RUNNING_LOCALLY:

    def sync_previous():
        print('sync_previous')
        from cnvrgv2 import Cnvrg
        cnvrg = Cnvrg()
        proj = cnvrg.projects.get("project_name")
        flow = proj.flows.get("flow_name")
        
        flow_versions = flow.flow_versions.list()   
        flow_version_title = flow_versions[0].title
        print({'flow_version_title':flow_version_title})
        fv = flow.flow_versions.get(flow_version_title)
            
        for i in range(len(fv.info()["fv_status"]["task_statuses"][0]["experiments"])):
            e = proj.experiments.get(fv.info()["fv_status"]["task_statuses"][0]["experiments"][i]["id"])
            e.pull_artifacts(wait_until_success=True, poll_interval=5)
        
        print('sync_previous done')

else:
    def sync_previous():
        raise NotImplementedError('sync is being called from local - oops')
