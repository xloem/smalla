import threadingj
import torch, accelerate
import tqdm


# module parallel kluge
# replicate() makes a copy of a model for every device
# parallel() then batches data across a submodule in parallel

def replicate(model):
    result = {
        'cpu': model.to(-1)
    }
    for dev in range(torch.cuda.device_count()):
        result[dev] = type(model)(model.config)
        result[dev].load_state_dict(model.state_dict())
    return result

def parallel(replicated, module_name, module_callback, **data, pbar_desc=None):
    class DictThread(threading.Thread):
        def __init__(self, target, dict_data, lock, key, *params, **kwparams):
            super().__init__(name=key)
            self.target = target
            self.dict_data = dict_data
            self.lock = lock
            self.params = params
            self.kwparams = kwparams
            self.start()
        def run(self):
            with self.lock:
                input = self.dict_data[key]
            output = self.target(key, input, *self.params, **self.kwparams)
            with self.lock:
                self.dict_data[key] = output
    def parallel_call(dict_data, transform, *params, **kwparams):
        lock = threading.Lock()
        threads = {
            key : DictThread(transform, dict_data, lock, key, *params, **kwparams)
            for key in dict_data.items()
        }
        for thread in threads.values():
            thread.join()
        return dict_data

    class DataSourceSink:
        def __init__(self, pbar, **data):
            # data is arbitrarily large and broken into keywords all with the same first dimension size
            self.data = data
            self.pbar = pbar
            self.next_idx = 0
            self.source_lock = threading.Lock()
            self.sink_lock = threading.Lock()
            self.unsourced = queue.Queue()
            self.names = list(self.data)
            self.total = len(self.data[self.names[0]])
            for name in self.names:
                assert len(self.data[name]) == self.total
            self.output = {}
            self.pbar.total = self.total
        def source(self, count):
            values = {name: [] for name in self.names}
            handle = []
            count_remaining = count
            while count_remaining > 0:
                try:
                    idx, ct, unsourced = self.unsourced.get(block=False)
                    if ct > count_remaining:
                        self.unsourced.put({
                            name: (
                                (idx + count_remaining, ct - count_remaining),
                                elem[count_remaining:]
                            )
                            for name, elem in unsourced.items()
                        })
                        ct = count_remaining
                    count_remaining -= ct
                    for name in self.names:
                        self.values[name].extend(unsourced[name][:ct])
                        assert len(self.values[name]) == ct
                    handle.append((idx, ct))
                except queue.Empty:
                    with self.source_lock:
                        if self.next_idx == self.total:
                            break
                        assert self.next_idx < self.total
                        idx = self.next_idx
                        next_idx = min(idx + count, self.total)
                        ct = next_idx - idx
                        self.next_idx = next_idx
                    count_remaining -= ct
                    for name in self.names:
                        self.values[name].extend(self.data[name][idx:next_idx])
                        assert len(self.values[name]) == ct
                    handle.append((idx, ct))
            return handle, values, count_remaining == 0
        def unsource(self, handle):
            for idx, ct in handle:
                self.unsourced.put({
                    name: (
                        (idx, ct),
                        elem[idx:idx+ct]
                    )
                    for name, elem in self.data.items()
                })
        def sink(self, handle, output):
            with self.sink_lock:
                if not len(self.output):
                    for key in output:
                        self.output[key] = [None] * self.total
            output_idx = 0
            for idx, ct in handle:
                for key in self.names:
                    self.output[key][idx:idx+ct] = output[key][output_idx:output_idx+ct]
                output_idx += ct
            with self.sink_lock:
                self.pbar.update(output_idx)

    class ModuleManager:
        def __init__(self, device, model, callback, data_source_sink):
            self.data_source_sink = data_source_sink
            self.module = model.named_modules()[module_name].to(device)
            if not hasattr(self.module, '__batch_bounds'):
                self.module.__batch_bounds = lambda: None
                self.module.__batch_bounds.lo = 1 # size known to work
                self.module.__batch_bounds.hi = len(batches+1) # size known to fail
            self.batch_bounds = self.module.__batch_bounds
            self.callback = callback
        def forward_with_batchsize_testing(self):
            bounds = self.batch_bounds
            while bounds.hi - bounds.lo > 1:
                test = (bounds.hi+bounds.lo)//2
                success, full = self.forward(test):
                if success:
                    if full:
                        bounds.lo = test
                    else:
                        return
                else:
                    bounds.hi = test
            while self.forward(bounds.lo)[1]:
                pass
        def forward(self, batch_size):
            handle, data, full = self.data_source_sink.source(test)
            try:
                output = self.callback(module, **data)
                self.data_source_sink.sink(handle, output)
                return True, full
            except:
                self.data_source_sink.unsource(handle)
                return False, full
        
    with tqdm.tqdm(desc=pbar_desc) as pbar:
        data_source_sink = DataSourceSink(data, pbar)
        module_managers = parallel_call(replicated, ModuleManager, module_callback, data_source_sink)
        parallel_call(module_managers, ModuleManager.forward_with_batchsize_testing)

    return data_source_sink.output
