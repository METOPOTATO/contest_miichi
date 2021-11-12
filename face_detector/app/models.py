from django.db import models

# Create your models here.
class People(models.Model):
    name = models.CharField(max_length=128, unique=True)
    age = models.IntegerField()
    job = models.CharField(max_length=128, null=True, blank=True)
    height = models.CharField(max_length=128, null=True, blank=True)
    description = models.CharField(max_length=128, null=True, blank=True)

    def __str__(self) -> str:
        return self.name