
import src.service.detector_service_pb2_grpc as servicer

class DetectorServicer(servicer.DetectorServiceServicer):
    
    def DetectImage(self, request, context):
        raise NotImplementedError()