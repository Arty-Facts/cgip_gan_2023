import torch
class TB_Profiler:
    def __init__(self, 
                base_dir=".",
                out_dir = "log", 
                wait=1, 
                warmup=1, 
                active=6, 
                repeat=2,
                record_shapes=True,
                profile_memory=True,
                with_stack=True
                ) -> None:
        self.prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{base_dir}/{out_dir}"),
            record_shapes=record_shapes,
            with_stack=with_stack, 
            profile_memory=profile_memory
            )
    def start(self):
        self.prof.start()

    def stop(self):
        self.prof.stop()

    def step(self):
        self.prof.step()